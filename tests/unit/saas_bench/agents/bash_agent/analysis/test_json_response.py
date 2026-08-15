"""Analysis JSON 回答的确定性格式归一化测试。"""

from json import JSONDecodeError

import pytest

from saas_bench.agents.bash_agent.analysis.json_response import parse_json_object


def test_parse_json_object_accepts_bare_and_single_fenced_json():
    expected = {"value": 1}

    assert parse_json_object('{"value": 1}') == expected
    assert parse_json_object('```json\n{"value": 1}\n```') == expected
    assert parse_json_object('```JSON\n{"value": 1}\n```\n') == expected


@pytest.mark.parametrize(
    "response",
    [
        '说明如下：\n```json\n{"value": 1}\n```',
        '```json\n{"value": 1}\n```\n补充说明',
        '```json\n{"value": 1}\n```\n```json\n{"value": 2}\n```',
    ],
)
def test_parse_json_object_rejects_text_outside_one_fence(response):
    with pytest.raises(JSONDecodeError):
        parse_json_object(response)


def test_parse_json_object_rejects_non_object_json():
    with pytest.raises(ValueError, match="top-level response"):
        parse_json_object("```json\n[]\n```")
