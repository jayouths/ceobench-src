"""Small JSON persistence helpers shared by the experiment runtime."""

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Replace a JSON file atomically so readers never observe partial content."""

    write_text_atomic(path, json.dumps(payload, indent=indent))


def write_text_atomic(path: Path, content: str) -> None:
    """原子替换文本文件，避免运行中断留下截断的过程产物。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 终态与断点恢复依赖这些产物；必须先完整写入临时文件，再原子切换。
    with open(temporary, "w") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
