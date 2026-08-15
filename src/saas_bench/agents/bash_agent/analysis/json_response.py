"""Analysis 模型 JSON 回答的最小格式归一化。"""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n(?P<payload>.*)\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def parse_json_object(text: str) -> dict[str, Any]:
    """接受裸 JSON 或单个完整 JSON 围栏，其他附加文本仍视为非法。"""

    match = _JSON_FENCE.fullmatch(text)
    payload_text = match.group("payload") if match else text
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("top-level response must be a JSON object")
    return payload
