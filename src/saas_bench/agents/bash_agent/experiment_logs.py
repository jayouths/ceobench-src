"""主实验轨迹与性能日志的统一写入入口。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LOG_FORMAT_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ExperimentLogWriter:
    """将原子过程与聚合指标分别写入两个 JSONL 文件。"""

    def __init__(
        self,
        *,
        run_id: str,
        trajectory_file: Path,
        performance_file: Path,
    ) -> None:
        self.run_id = run_id
        self.trajectory_file = trajectory_file
        self.performance_file = performance_file

    @staticmethod
    def week_index(sim_day: int) -> int:
        """Day 0 属于第 0 周，Day 7 属于第 1 周。"""
        return sim_day // 7

    def trajectory(self, event_type: str, sim_day: int, **fields: Any) -> dict[str, Any]:
        entry = self._entry(event_type, sim_day, fields)
        self._append(self.trajectory_file, entry)
        return entry

    def performance(self, event_type: str, sim_day: int, **fields: Any) -> dict[str, Any]:
        entry = self._entry(event_type, sim_day, fields)
        self._append(self.performance_file, entry)
        return entry

    def has_trajectory_event(self, event_type: str, sim_day: int) -> bool:
        return any(
            event.get("event_type") == event_type and event.get("sim_day") == sim_day
            for event in self.read_trajectory()
        )

    def has_performance_event(self, event_type: str, sim_day: int) -> bool:
        return any(
            event.get("event_type") == event_type and event.get("sim_day") == sim_day
            for event in self._read_jsonl(self.performance_file)
        )

    def read_trajectory(self) -> Iterable[dict[str, Any]]:
        return self._read_jsonl(self.trajectory_file)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        events = []
        with open(path) as file:
            for line in file:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def summarize_week(self, sim_day: int) -> dict[str, Any]:
        """从原子轨迹重算周级指标，恢复实验时也不会漏掉旧批次。"""
        modules: dict[str, dict[str, Any]] = {}
        tool_summary = {
            "call_count": 0,
            "completed_count": 0,
            "error_count": 0,
            "elapsed_seconds": 0.0,
        }
        dashboard_seconds = 0.0

        for event in self.read_trajectory():
            if event.get("sim_day") != sim_day:
                continue
            if event.get("event_type") == "dashboard":
                dashboard_seconds += float(event.get("elapsed_seconds", 0.0))
                continue
            if event.get("event_type") == "tool_execution":
                tool_summary["call_count"] += 1
                tool_summary["elapsed_seconds"] += float(
                    event.get("elapsed_seconds", 0.0)
                )
                if event.get("status") == "completed":
                    tool_summary["completed_count"] += 1
                else:
                    tool_summary["error_count"] += 1
                continue
            if event.get("event_type") != "llm_call":
                continue

            component = str(event.get("component") or "unknown")
            summary = modules.setdefault(
                component,
                {
                    "call_count": 0,
                    "completed_count": 0,
                    "accepted_count": 0,
                    "invalid_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "elapsed_seconds": 0.0,
                    "cost_by_currency": {},
                },
            )
            summary["call_count"] += 1
            status = event.get("status")
            if status in {"valid", "invalid", "completed"}:
                summary["completed_count"] += 1
            if status == "valid":
                summary["accepted_count"] += 1
            elif status == "invalid":
                summary["invalid_count"] += 1
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ):
                summary[field] += int(event.get(field, 0))
            summary["elapsed_seconds"] += float(event.get("elapsed_seconds", 0.0))
            currency = event.get("currency")
            if currency:
                costs = summary["cost_by_currency"]
                costs[currency] = costs.get(currency, 0.0) + float(
                    event.get("cost_amount", 0.0)
                )

        tool_summary["elapsed_seconds"] = round(
            tool_summary["elapsed_seconds"], 6
        )
        dashboard_seconds = round(dashboard_seconds, 6)
        for summary in modules.values():
            summary["elapsed_seconds"] = round(summary["elapsed_seconds"], 6)
        return {
            "modules": modules,
            "tools": tool_summary,
            "dashboard_seconds": dashboard_seconds,
        }

    def _entry(
        self, event_type: str, sim_day: int, fields: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "format_version": LOG_FORMAT_VERSION,
            "timestamp": _now(),
            "run_id": self.run_id,
            "event_type": event_type,
            "sim_day": sim_day,
            "week_index": self.week_index(sim_day),
            **fields,
        }

    @staticmethod
    def _append(path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
