"""只读投影宿主请求日志，不向 Agent 暴露执行历史或原始回执。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _boundary(value: Any, offsets: set[int], name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value not in offsets:
        raise ValueError(f"Invalid {name} request journal boundary: {value!r}")
    return value


def _read_actions(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    actions: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    offsets = {0}
    if not path.exists():
        return actions, offsets
    with path.open("rb") as file:
        while line := file.readline():
            end = file.tell()
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete request journal line at byte {end}")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid request journal line at byte {end}") from exc
            if (not isinstance(record, dict)
                    or not isinstance(record.get("request_id"), str)
                    or not isinstance(record.get("digest"), str)):
                raise ValueError(f"Invalid request journal record at byte {end}")
            if record.get("status") == "pending" and pending is None:
                if any(action["request_id"] == record["request_id"] for action in actions):
                    raise ValueError(f"Duplicate request ID at byte {end}")
                pending = record
                actions.append({
                    "request_id": record["request_id"],
                    "operation": record.get("operation"),
                    "input": record.get("input"),
                    "day_before": record.get("day_before"),
                    "day_after": None,
                    "started_at": record.get("started_at"),
                    "completed_at": None,
                    "status": "pending",
                    "success": None,
                    "_end": end,
                })
            elif record.get("status") == "completed" and pending is not None:
                if (record["request_id"] != pending["request_id"]
                        or record.get("digest") != pending.get("digest")):
                    raise ValueError(f"Mismatched request completion at byte {end}")
                action = actions[-1]
                action.update(
                    day_after=record.get("day_after"),
                    completed_at=record.get("completed_at"),
                    status="completed",
                    success=record.get("success"),
                    _end=end,
                )
                pending = None
            else:
                raise ValueError(f"Invalid request journal order at byte {end}")
            offsets.add(end)
    return actions, offsets


def audit_run(run_dir: Path | str) -> dict[str, Any]:
    """按字节边界区分执行完成、世界已落盘和 Runner 已确认。"""
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = _read_json(checkpoint_path) if checkpoint_path.exists() else None
    if checkpoint is not None:
        session_id = checkpoint["session_id"]
        if not isinstance(session_id, str) or Path(session_id).name != session_id:
            raise ValueError("Invalid checkpoint session ID")
        session_dir = run_dir / "agent_workspace" / "sessions" / session_id
    else:
        sessions = list((run_dir / "agent_workspace" / "sessions").glob("*/session.json"))
        if len(sessions) != 1:
            raise ValueError("Run must have a checkpoint or exactly one session")
        session_dir = sessions[0].parent
        session_id = session_dir.name

    metadata = _read_json(session_dir / "session.json")
    if metadata.get("session_id") not in (None, session_id):
        raise ValueError("Session metadata belongs to another session")
    actions, offsets = _read_actions(session_dir / "request-journal.jsonl")
    world_bytes = _boundary(metadata.get("request_journal_bytes"), offsets, "world snapshot")
    runner_bytes = _boundary(
        (checkpoint.get("runtime") or {}).get("request_journal_bytes")
        if checkpoint else None,
        offsets, "runner checkpoint",
    )
    if world_bytes is not None and runner_bytes is not None and runner_bytes > world_bytes:
        raise ValueError("Runner checkpoint exceeds the persisted world boundary")
    for action in actions:
        end = action.pop("_end")
        # pending 只证明请求被接收；即使日志落盘，也不能推断世界动作已完成。
        action["world_snapshot_covered"] = (
            end <= world_bytes if world_bytes is not None and action["status"] == "completed"
            else False if world_bytes is not None else None
        )
        action["runner_checkpoint_covered"] = (
            end <= runner_bytes if runner_bytes is not None and action["status"] == "completed"
            else False if runner_bytes is not None else None
        )
    return {
        "session_id": session_id,
        "world_snapshot_bytes": world_bytes,
        "runner_checkpoint_bytes": runner_bytes,
        "actions": actions,
    }
