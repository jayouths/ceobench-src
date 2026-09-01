"""Bash Agent 断点文件的保存、校验与恢复。"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.workspace import AgentWorkspaceRepository
from saas_bench.experiment.json_io import write_json_atomic


@dataclass(frozen=True)
class CheckpointRestorePlan:
    """断点预检后，恢复 Agent 所需的内存状态。"""

    session_id: str
    conversation_payload: dict[str, Any]


class CheckpointStore:
    """管理单次实验目录中的可信断点及其附属文件。"""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        agent_workspace: Path,
        trajectory_log_file: Path,
        performance_log_file: Path,
        workspace_repository: AgentWorkspaceRepository,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.agent_workspace = Path(agent_workspace)
        self.trajectory_log_file = Path(trajectory_log_file)
        self.performance_log_file = Path(performance_log_file)
        self.workspace_repository = workspace_repository

    def save(
        self,
        *,
        day: int,
        cash: float,
        session_id: str,
        environment_llm_usage: dict[str, Any],
        analysis_usage: dict[str, Any],
        agent: BashAgent | None,
        decision_cost_by_currency: dict[str, float],
        server_log_offsets: dict[str, int],
        resume_conversation: bool = False,
        pending_observation: str | None = None,
    ) -> dict[str, Any]:
        """保存不可变运行状态，最后原子切换 checkpoint.json。"""
        session_nmdb = self.agent_workspace / "sessions" / session_id / "world.nmdb"
        if not session_nmdb.is_file():
            raise FileNotFoundError(f"Persisted session database not found: {session_nmdb}")

        # 数据库和对话均先写入唯一版本文件。写入中断时，旧 checkpoint 仍然有效。
        checkpoint_id = uuid.uuid4().hex
        checkpoint_db_dir = self.workspace_dir / ".checkpoint_dbs"
        checkpoint_db_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_db = checkpoint_db_dir / f"world_day_{day}_{checkpoint_id}.nmdb"
        checkpoint_db_tmp = checkpoint_db.with_suffix(".nmdb.tmp")
        shutil.copy2(session_nmdb, checkpoint_db_tmp)
        os.replace(checkpoint_db_tmp, checkpoint_db)

        runtime_dir = self.workspace_dir / ".checkpoint_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        conversation_snapshot = runtime_dir / f"conversation_{checkpoint_id}.json"
        if agent is None:
            write_json_atomic(
                conversation_snapshot,
                {
                    "resume_conversation": False,
                    "conversation": [],
                    "current_day": 0,
                    "turns_today": 0,
                    "total_turns": 0,
                },
            )
        else:
            agent.save_checkpoint_snapshot(
                conversation_snapshot,
                resume_conversation=resume_conversation,
                pending_observation=pending_observation,
            )

        workspace_commit = self.workspace_repository.capture_checkpoint_commit(day)
        checkpoint = {
            "day": day,
            "cash": float(cash),
            "session_id": session_id,
            "database": {
                "file": str(checkpoint_db.relative_to(self.workspace_dir)),
            },
            "runtime": {
                "runner_log_offsets": self.capture_runner_log_offsets(),
                "server_log_offsets": dict(server_log_offsets),
                "conversation": {
                    "file": str(conversation_snapshot.relative_to(self.workspace_dir)),
                },
                "workspace_commit": workspace_commit,
                "environment_llm": environment_llm_usage,
                "analysis": analysis_usage,
                "agent": {
                    "total_turns": agent.total_turns if agent else 0,
                    "input_tokens": agent.total_input_tokens if agent else 0,
                    "output_tokens": agent.total_output_tokens if agent else 0,
                    "cached_tokens": agent.total_cached_tokens if agent else 0,
                    "reasoning_tokens": agent.total_reasoning_tokens if agent else 0,
                    "decision_cost_by_currency": dict(decision_cost_by_currency),
                },
            },
        }
        write_json_atomic(self.workspace_dir / "checkpoint.json", checkpoint)

        # 根目录副本只用于人工分析；恢复始终读取 checkpoint 指向的不可变文件。
        try:
            latest_database = self.workspace_dir / "world.nmdb"
            latest_database_tmp = latest_database.with_suffix(".nmdb.tmp")
            shutil.copy2(checkpoint_db, latest_database_tmp)
            os.replace(latest_database_tmp, latest_database)
        except OSError:
            pass

        self._remove_stale_files(checkpoint_db, conversation_snapshot)
        return checkpoint

    def load(self) -> dict[str, Any] | None:
        """读取并校验恢复所需的断点结构。"""
        checkpoint_file = self.workspace_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None
        with open(checkpoint_file) as file:
            checkpoint = self.validate(json.load(file))
        if not (self.workspace_dir / "config.json").is_file():
            raise FileNotFoundError("Checkpoint run config is missing")
        return checkpoint

    def preflight(self, checkpoint: dict[str, Any]) -> CheckpointRestorePlan:
        """只读校验所有持久化产物，全部通过后才允许修改当前状态。"""
        runtime = checkpoint["runtime"]
        database_path = self.artifact_path(checkpoint["database"]["file"], "database")
        if not database_path.is_file():
            raise FileNotFoundError(f"Checkpoint database not found: {database_path}")

        conversation_path = self.artifact_path(
            runtime["conversation"]["file"], "conversation"
        )
        if not conversation_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint conversation not found: {conversation_path}"
            )
        conversation_payload = BashAgent.parse_checkpoint_snapshot(conversation_path)

        session_id = checkpoint["session_id"]
        session_dir = self.agent_workspace / "sessions" / session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint session directory not found: {session_dir}")
        session_meta = session_dir / "session.json"
        if not session_meta.is_file():
            raise FileNotFoundError(
                f"Checkpoint session metadata not found: {session_meta}"
            )
        try:
            metadata = json.loads(session_meta.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Checkpoint session metadata is invalid JSON") from exc
        if metadata.get("session_id") not in {None, session_id}:
            raise ValueError("Checkpoint session metadata belongs to another session")

        commit = runtime["workspace_commit"]
        if not self.workspace_repository.commit_exists(commit):
            raise ValueError(f"Agent workspace checkpoint commit does not exist: {commit}")

        runner_offsets = self.validate_runner_log_offsets(
            runtime["runner_log_offsets"]
        )
        self.validate_offsets_within_files(
            runner_offsets, self.runner_log_files(), "runner log"
        )
        server_files = self.server_log_files(session_id)
        server_offsets = self.validate_server_log_offsets_for_files(
            runtime["server_log_offsets"], server_files
        )
        self.validate_offsets_within_files(
            server_offsets, server_files, "server log"
        )
        return CheckpointRestorePlan(
            session_id=session_id,
            conversation_payload=conversation_payload,
        )

    def restore_runner_logs(self, offsets: dict[str, Any]) -> None:
        validated = self.validate_runner_log_offsets(offsets)
        self._truncate_files(self.runner_log_files(), validated, "runner log")

    def restore_server_logs(
        self, offsets: Any, session_id: str
    ) -> None:
        """在 EventLogger 打开追加文件前，回退模拟器日志。"""
        files = self.server_log_files(session_id)
        validated = self.validate_server_log_offsets_for_files(offsets, files)
        self._truncate_files(files, validated, "server log")
        # _meta.json 代表实验终态，恢复断点后已失效。
        event_log = files["event_log"]
        event_log.with_name(event_log.stem + "_meta.json").unlink(missing_ok=True)

    def restore_database(self, checkpoint: dict[str, Any]) -> None:
        """在模拟器启动前恢复数据库和会话日期。"""
        day = checkpoint["day"]
        session_id = checkpoint["session_id"]
        checkpoint_database = self.artifact_path(
            checkpoint["database"]["file"], "database"
        )
        session_dir = self.agent_workspace / "sessions" / session_id
        session_database = session_dir / "world.nmdb"
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint session directory not found: {session_dir}")
        shutil.copy2(checkpoint_database, session_database)
        print(f"  Restored DB from checkpoint (day {day})")

        session_meta = session_dir / "session.json"
        if not session_meta.is_file():
            raise FileNotFoundError(
                f"Checkpoint session metadata not found: {session_meta}"
            )
        metadata = json.loads(session_meta.read_text())
        metadata["current_day"] = day
        metadata["status"] = "created"
        metadata.pop("port", None)
        metadata.pop("pid", None)
        write_json_atomic(session_meta, metadata)

    def validate(self, checkpoint: Any) -> dict[str, Any]:
        """只校验实际恢复路径需要的字段。"""
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint root must be an object")
        required_root = {"day", "cash", "session_id", "database", "runtime"}
        missing = required_root - checkpoint.keys()
        if missing:
            raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
        self._require_non_negative_integer(checkpoint["day"], "day")
        self._require_finite_number(checkpoint["cash"], "cash")
        if not isinstance(checkpoint["session_id"], str) or not checkpoint["session_id"]:
            raise ValueError("Checkpoint session_id must be a non-empty string")

        database = checkpoint["database"]
        if not isinstance(database, dict):
            raise ValueError("Checkpoint database must be an object")
        if not isinstance(database.get("file"), str) or not database["file"]:
            raise ValueError("Checkpoint database file must be a non-empty string")

        runtime = checkpoint["runtime"]
        required_runtime = {
            "runner_log_offsets",
            "server_log_offsets",
            "conversation",
            "workspace_commit",
            "environment_llm",
            "analysis",
            "agent",
        }
        if not isinstance(runtime, dict):
            raise ValueError("Checkpoint runtime must be an object")
        missing = required_runtime - runtime.keys()
        if missing:
            raise ValueError(f"Checkpoint runtime is missing fields: {sorted(missing)}")
        self.validate_runner_log_offsets(runtime["runner_log_offsets"])
        self.validate_server_log_offsets(
            runtime["server_log_offsets"], checkpoint["session_id"]
        )
        if not isinstance(runtime["workspace_commit"], str) or not runtime["workspace_commit"]:
            raise ValueError("Checkpoint workspace_commit must be a non-empty string")
        if not isinstance(runtime["environment_llm"], dict):
            raise ValueError("Checkpoint environment_llm must be an object")

        analysis = runtime["analysis"]
        if not isinstance(analysis, dict):
            raise ValueError("Checkpoint analysis must be an object")
        for field in ("role_report_days", "state_portrait_days"):
            if not isinstance(analysis.get(field), list):
                raise ValueError(f"Checkpoint analysis {field} must be a list")

        conversation = runtime["conversation"]
        if not isinstance(conversation, dict):
            raise ValueError("Checkpoint conversation must be an object")
        if not isinstance(conversation.get("file"), str) or not conversation["file"]:
            raise ValueError("Checkpoint conversation file must be a non-empty string")

        agent = runtime["agent"]
        required_agent = {
            "total_turns",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "decision_cost_by_currency",
        }
        if not isinstance(agent, dict):
            raise ValueError("Checkpoint agent must be an object")
        missing = required_agent - agent.keys()
        if missing:
            raise ValueError(f"Checkpoint agent is missing fields: {sorted(missing)}")
        return checkpoint

    def runner_log_files(self) -> dict[str, Path]:
        return {
            "trajectory": self.trajectory_log_file,
            "performance": self.performance_log_file,
        }

    def capture_runner_log_offsets(self) -> dict[str, int]:
        return {
            name: path.stat().st_size if path.exists() else 0
            for name, path in self.runner_log_files().items()
        }

    def validate_runner_log_offsets(self, offsets: Any) -> dict[str, int]:
        expected_names = set(self.runner_log_files())
        if not isinstance(offsets, dict) or set(offsets) != expected_names:
            raise ValueError(
                f"Checkpoint log offsets must contain exactly: {sorted(expected_names)}"
            )
        self._validate_non_negative_offsets(offsets, "checkpoint log")
        return dict(offsets)

    def server_log_files(self, session_id: str) -> dict[str, Path]:
        if not session_id:
            raise ValueError("session_id is required to resolve server logs")
        session_dir = self.agent_workspace / "sessions" / session_id
        return {
            "history": session_dir / "history.jsonl",
            "event_log": session_dir / "logs" / f"run_{session_id}.jsonl",
        }

    def validate_server_log_offsets(
        self, offsets: Any, session_id: str
    ) -> dict[str, int]:
        return self.validate_server_log_offsets_for_files(
            offsets, self.server_log_files(session_id)
        )

    @staticmethod
    def validate_server_log_offsets_for_files(
        offsets: Any, files: dict[str, Path]
    ) -> dict[str, int]:
        expected_names = set(files)
        if not isinstance(offsets, dict) or set(offsets) != expected_names:
            raise ValueError(
                f"Checkpoint server log offsets must contain exactly: {sorted(expected_names)}"
            )
        CheckpointStore._validate_non_negative_offsets(offsets, "checkpoint server log")
        return dict(offsets)

    @staticmethod
    def validate_offsets_within_files(
        offsets: dict[str, int], files: dict[str, Path], label: str
    ) -> None:
        for name, path in files.items():
            current_size = path.stat().st_size if path.exists() else 0
            if offsets[name] > current_size:
                raise ValueError(
                    f"Checkpoint {label} offset for {name} exceeds file size: "
                    f"{offsets[name]} > {current_size}"
                )

    def artifact_path(self, relative_path: str, label: str) -> Path:
        path = (self.workspace_dir / relative_path).resolve()
        workspace_root = self.workspace_dir.resolve()
        if workspace_root not in path.parents:
            raise ValueError(
                f"Checkpoint {label} path escapes run directory: {relative_path}"
            )
        return path

    def _remove_stale_files(
        self, current_database: Path, current_conversation: Path
    ) -> None:
        # 清理失败最多留下无引用文件，不影响刚提交的可信断点。
        for stale_database in current_database.parent.glob("*.nmdb"):
            if stale_database != current_database:
                try:
                    stale_database.unlink(missing_ok=True)
                except OSError:
                    pass
        for stale_conversation in current_conversation.parent.glob(
            "conversation_*.json"
        ):
            if stale_conversation != current_conversation:
                try:
                    stale_conversation.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _truncate_files(
        files: dict[str, Path], offsets: dict[str, int], label: str
    ) -> None:
        CheckpointStore.validate_offsets_within_files(offsets, files, label)
        for name, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a+b") as file:
                file.truncate(offsets[name])

    @staticmethod
    def _validate_non_negative_offsets(offsets: dict[str, Any], label: str) -> None:
        for name, offset in offsets.items():
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError(f"Invalid {label} offset for {name}: {offset!r}")

    @staticmethod
    def _require_finite_number(value: Any, field: str) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        return float(value)

    @staticmethod
    def _require_non_negative_integer(value: Any, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        return value
