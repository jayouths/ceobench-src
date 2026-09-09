"""Content-addressed cache for deterministic environment LLM responses."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .llm_provider import TextLLMResult


_SCHEMA_VERSION = 1
_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LLMResponseCache:
    """Share one canonical response for each complete LLM request."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory).expanduser().resolve()

    def get_or_create(
        self,
        request: Mapping[str, Any],
        create: Callable[[], TextLLMResult],
    ) -> TextLLMResult:
        """Return the cached response or atomically create the first response."""
        request_json = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        entry_path = self.directory / request_sha256[:2] / f"{request_sha256}.json"
        lock_path = self.directory / "locks" / f"{request_sha256}.lock"
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # 线程锁解决同进程并发，文件锁解决 Baseline/Analysis 跨进程并发。
        thread_lock = _thread_lock(lock_path)
        with thread_lock, lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if entry_path.exists():
                return self._read(entry_path, request_sha256)

            response = create()
            self._write(entry_path, request_sha256, response)
            return response

    @staticmethod
    def _read(entry_path: Path, request_sha256: str) -> TextLLMResult:
        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid LLM response cache entry: {entry_path}") from exc
        if (
            data.get("schema_version") != _SCHEMA_VERSION
            or data.get("request_sha256") != request_sha256
            or not isinstance(data.get("text"), str)
            or not data["text"]
            or not isinstance(data.get("model"), str)
            or not data["model"]
        ):
            raise RuntimeError(f"Invalid LLM response cache entry: {entry_path}")
        # 本地复用没有向 Provider 发起请求，Token 和实际费用均记为零。
        return TextLLMResult(
            text=data["text"],
            model=data["model"],
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=None,
            raw_response=None,
            response_cache_hit=True,
        )

    @staticmethod
    def _write(
        entry_path: Path,
        request_sha256: str,
        response: TextLLMResult,
    ) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "text": response.text,
            "model": response.model,
            "origin_usage": {
                "input_tokens": response.input_tokens,
                "cached_tokens": response.cached_tokens,
                "output_tokens": response.output_tokens,
                "reasoning_tokens": response.reasoning_tokens,
            },
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{entry_path.name}.",
            suffix=".tmp",
            dir=entry_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, entry_path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _thread_lock(path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(path, threading.Lock())
