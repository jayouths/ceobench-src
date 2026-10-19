"""真实源码/公开运行包的进程级恢复验证，不调用任何模型。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from saas_bench.novamind_api._transport import request_json


@pytest.mark.slow
@pytest.mark.parametrize("packaged", [False, True], ids=["source", "zipapp"])
def test_process_resume_preserves_actions_and_crash_rejects_rollback(tmp_path, monkeypatch, packaged):
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.pop("NOVAMIND_API_SOCKET", None)
    env.pop("NOVAMIND_API_PORT", None)
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONHASHSEED"] = "0"
    env["NMDB_KEY"] = "process-regression-test-key"
    env["CEOBENCH_SIMULATOR_LLM_CONFIG"] = json.dumps({
        "social_post_llm_api_type": "openai_chat_completions",
        "social_post_llm_model": "DeepSeek-V4-Flash",
        "social_post_llm_base_url": "http://127.0.0.1:1/v1",
        "social_post_llm_api_key_required": False,
        "social_post_llm_max_tokens": 64,
        "social_post_llm_response_cache_path": str(tmp_path / "model-cache"),
        "social_post_llm_pricing": {"DeepSeek-V4-Flash": {"currency": "CNY"}},
    })
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    prefix = [sys.executable]
    if packaged:
        prefix.append(str(root / "public" / "novamind-operation"))
        env["NOVAMIND_SERVER_MODE"] = "1"
    else:
        prefix += ["-m", "saas_bench.runtime.server_entry"]
        env.pop("NOVAMIND_SERVER_MODE", None)
    command = prefix + ["--base", str(tmp_path)]
    processes = []

    def invoke(*args):
        return subprocess.run(command + list(args), env=env, cwd=root,
                              capture_output=True, text=True, timeout=30)

    created = invoke("new-session", "--seed", "42", "--days", "14")
    assert created.returncode == 0, created.stderr
    session_id = json.loads(created.stdout)["session_id"]
    session_dir = tmp_path / "sessions" / session_id
    port_path = session_dir / ".server.port"
    world = session_dir / "world.nmdb"

    def start():
        with (tmp_path / "server-test.log").open("ab") as log:
            process = subprocess.Popen(command + ["start-server", "--session", session_id],
                                       env=env, cwd=root, stdout=log, stderr=log)
        processes.append(process)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path / "server-test.log").read_text()
            if port_path.exists():
                port = int(port_path.read_text())
                if request_json("GET", "/health", port=port, timeout=3)["status"] == "ok":
                    return process, port
            time.sleep(0.05)
        pytest.fail("Test server did not start")

    def call(port, request_id, tool, args):
        return request_json("POST", "/call", {
            "request_id": request_id, "tool": tool, "args": args,
        }, port=port, timeout=5)

    def stop(process):
        process.terminate()
        assert process.wait(timeout=15) == 0
        assert not port_path.exists()

    try:
        process, port = start()
        duplicate = invoke("start-server", "--session", session_id)
        assert duplicate.returncode != 0 and "already has a running server" in duplicate.stderr
        assert call(port, "price", "set_prices", {"A": 19})["success"]
        ops = call(port, "ops", "set_targeted_ops_spend", {"by_group": {"S1": 17}})
        assert ops["success"]
        persisted = request_json("POST", "/persist-checkpoint", {"expected_day": 0}, port=port, timeout=10)
        assert persisted["success"] and persisted["environment_llm_usage"]["cost_by_currency"] == {}
        stop(process)

        process, port = start()
        before_bytes = (session_dir / "request-journal.jsonl").stat().st_size
        assert call(port, "ops", "set_targeted_ops_spend", {"by_group": {"S1": 17}}) == ops
        assert (session_dir / "request-journal.jsonl").stat().st_size == before_bytes
        assert call(port, "inspect", "set_targeted_ops_spend", {})["data"]["by_group"] == {"S1": 17.0}
        assert request_json("POST", "/query", {
            "sql": "SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1",
        }, port=port)["rows"] == [{"price_A": 19.0}]
        assert request_json("GET", "/request-status/price", port=port)["status"] == "completed"
        stop(process)

        saved_world = world.read_bytes()
        process, port = start()
        assert call(port, "uncheckpointed", "set_prices", {"A": 99})["success"]
        # 已确认动作没有进入世界快照时，硬中断不能通过重启退回旧世界。
        process.kill()
        process.wait(timeout=5)
        failed = invoke("start-server", "--session", session_id)
        assert failed.returncode != 0 and "request log exceeds persisted world" in failed.stderr
        assert world.read_bytes() == saved_world
        denied = invoke("new-session", "--seed", "42", "--days", "14")
        assert json.loads(denied.stdout)["error"] == "one_game_only"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
