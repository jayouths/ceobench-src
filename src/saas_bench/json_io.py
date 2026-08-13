"""Small JSON persistence helpers shared by the experiment runtime."""

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Replace a JSON file atomically so readers never observe partial content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 终态与断点恢复都依赖这些 JSON；必须先完整写入临时文件，再原子切换。
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=indent)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
