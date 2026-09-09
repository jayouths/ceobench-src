"""Deterministic random generators for semantic simulator events."""

import hashlib
import json

from numpy.random import Generator, default_rng


def event_rng(seed: int, event_type: str, *identity: int | str) -> Generator:
    """Create a deterministic RNG for one semantic event.

    The result depends on event identity rather than execution order, so an Agent
    action cannot advance the simulator's main random stream.
    """
    # 使用稳定的密码学哈希，避免 Python hash() 的进程级随机化。
    payload = json.dumps(
        [seed, event_type, *identity],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return default_rng(derived_seed)
