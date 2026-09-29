"""Durable request identities for state-changing simulator API calls."""

import hashlib
import json
import os
from pathlib import Path


class RequestJournal:
    """Append-only host-side log; a database snapshot records its exact boundary."""

    def __init__(self, path: Path, persisted_bytes: int | None):
        self.path = Path(path)
        self.records = {}
        size = self.path.stat().st_size if self.path.exists() else 0
        # A completed reply is recoverable only when the matching world snapshot
        # was durably saved. Never silently roll a request back on server restart.
        if size != (persisted_bytes or 0):
            raise RuntimeError("unrecoverable_session: request log exceeds persisted world")
        if self.path.exists():
            with self.path.open("rb") as file:
                for line in file:
                    record = json.loads(line)
                    self.records[record["request_id"]] = record
        if any(record["status"] != "completed" for record in self.records.values()):
            raise RuntimeError("unrecoverable_session: interrupted request")

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def lookup(self, request_id: str) -> dict:
        record = self.records.get(request_id)
        if record is None:
            return {"request_id": request_id, "status": "unknown"}
        return {
            "request_id": request_id,
            "status": record["status"],
            "result": record.get("result"),
        }

    def run(self, request_id: str, operation: str, payload: dict, callback):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            return {"success": False, "error": "request_id_required"}
        digest = hashlib.sha256(json.dumps(
            [operation, payload], sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode()).hexdigest()
        previous = self.records.get(request_id)
        if previous:
            if previous["digest"] != digest:
                return {"success": False, "error": "request_id_payload_mismatch"}
            if previous["status"] == "completed":
                return previous["result"]
            return {"success": False, "error": "unrecoverable_session"}
        if any(record["status"] != "completed" for record in self.records.values()):
            return {"success": False, "error": "unrecoverable_session"}

        self._append({
            "request_id": request_id, "digest": digest, "status": "pending",
        })
        # Any exception, including a timed-out week, leaves pending on disk.
        # The world may have changed; replaying the action would be unsafe.
        result = callback()
        if result.get("error") in {"step_week_timeout", "step_day_timeout"}:
            return result
        self._append({
            "request_id": request_id, "digest": digest, "status": "completed",
            "result": result,
        })
        return result

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        creating = not self.path.exists()
        data = (json.dumps(record, default=str, ensure_ascii=False) + "\n").encode()
        with self.path.open("ab") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if creating:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        self.records[record["request_id"]] = record
