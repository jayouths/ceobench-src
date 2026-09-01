"""Bash Agent 使用的模拟器服务进程与 HTTP 通信。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class SimulatorServer:
    """管理 host 侧模拟器进程，不包含经营规则或实验决策。"""

    def __init__(
        self,
        *,
        run_id: str,
        agent_workspace: Path,
        logs_dir: Path,
        public_dir: Path,
        simulator_llm_config: dict[str, Any],
        env_vars: dict[str, str],
    ) -> None:
        self.run_id = run_id
        self.agent_workspace = Path(agent_workspace)
        self.logs_dir = Path(logs_dir)
        self.public_dir = Path(public_dir)
        self.simulator_llm_config = dict(simulator_llm_config)
        self.env_vars = dict(env_vars)
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.socket_dir: Path | None = None
        self.api_socket_path: Path | None = None
        self.stderr_file = None

    def create_session(
        self,
        *,
        source_cli: Path,
        total_days: int,
        seed: int,
        initial_cash: float,
        scenario: str,
    ) -> str:
        """通过 host 侧公开 CLI 创建模拟器会话。"""
        result = subprocess.run(
            [
                sys.executable,
                str(source_cli),
                "--base",
                str(self.agent_workspace),
                "new-session",
                "--days",
                str(total_days),
                "--seed",
                str(seed),
                "--cash",
                str(initial_cash),
                "--scenario",
                scenario,
            ],
            capture_output=True,
            text=True,
            env=self.environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"novamind-operation new-session failed:\n{result.stderr}\n{result.stdout}"
            )
        session_info = json.loads(result.stdout)
        session_id = session_info.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(f"Simulator returned invalid session: {session_info!r}")
        return session_id

    def start(self, session_id: str) -> None:
        """启动模拟器进程，并等待健康检查通过。"""
        zipapp_path = self.public_dir / "novamind-operation"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stderr_path = self.logs_dir / "api_server_stderr.log"
        # stderr 必须写文件。若使用无人读取的 PIPE，大量异常输出会堵塞子进程。
        self.stderr_file = open(stderr_path, "ab", buffering=0)
        # 短路径避免 Unix Socket 的系统长度限制，且不放入 Agent 可写目录。
        self.socket_dir = Path(tempfile.mkdtemp(prefix=f"ceobench-{self.run_id}-"))
        self.api_socket_path = self.socket_dir / "api.sock"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(zipapp_path),
                "--base",
                str(self.agent_workspace),
                "start-server",
                "--session",
                session_id,
                "--unix-socket",
                str(self.api_socket_path),
            ],
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            env=self.environment(),
        )

        first_line = self.process.stdout.readline()
        if not first_line:
            try:
                stderr_tail = stderr_path.read_bytes()[-4096:]
            except OSError:
                stderr_tail = b"<stderr log unavailable>"
            raise RuntimeError(
                f"Server failed to start:\n{stderr_tail.decode(errors='replace')}"
            )

        server_info = json.loads(first_line)
        self.port = server_info["port"]
        if server_info.get("unix_socket") != str(self.api_socket_path):
            raise RuntimeError("Server did not expose the requested Agent API socket")
        print(f"  Server started: port={self.port}, pid={server_info['pid']}")

        for _ in range(60):
            try:
                self.get("/health", timeout=2)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Server did not respond to /health after 30s")

    def stop(self) -> None:
        """幂等关闭完整或部分启动的模拟器进程。"""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=210)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.port = None
        if self.socket_dir is not None:
            shutil.rmtree(self.socket_dir, ignore_errors=True)
        self.socket_dir = None
        self.api_socket_path = None
        if self.stderr_file is not None:
            try:
                self.stderr_file.close()
            except OSError:
                pass
            self.stderr_file = None

    def get(self, path: str, timeout: float = 30) -> dict[str, Any]:
        request = urllib.request.Request(self.url(path))
        response = urllib.request.urlopen(request, timeout=timeout)
        return json.loads(response.read())

    def post(
        self,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: float = 1800,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url(path),
            data=json.dumps(data or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # 业务校验错误同样返回 JSON，保留服务端给出的明确原因。
            try:
                return json.loads(exc.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise

    def url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("Simulator server is not running")
        return f"http://127.0.0.1:{self.port}{path}"

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["NOVAMIND_SERVER_MODE"] = "1"
        if self.simulator_llm_config:
            env["CEOBENCH_SIMULATOR_LLM_CONFIG"] = json.dumps(
                self.simulator_llm_config, separators=(",", ":")
            )
            for field, value in self.simulator_llm_config.items():
                if (
                    field.endswith("_api_key_env")
                    and value
                    and value in self.env_vars
                ):
                    env.setdefault(value, self.env_vars[value])
        return env
