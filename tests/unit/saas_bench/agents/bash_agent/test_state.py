"""Bash Agent 对应逻辑的快速单元测试。"""

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner


def test_game_status_is_the_authoritative_day_source():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._http_get = lambda path: {
        "day": 7,
        "cash": 900_000,
        "subscribers": 10,
        "timed_out": False,
    }

    assert runner._get_game_status()["day"] == 7

@pytest.mark.parametrize("status", [{}, {"day": None}, {"day": "7"}, {"day": -1}])
def test_invalid_game_status_fails_instead_of_falling_back_to_day_zero(status):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._http_get = lambda path: status

    with pytest.raises(RuntimeError, match="Invalid simulator status"):
        runner._get_game_status()

def test_tool_execution_does_not_parse_dashboard_text():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.tool_executor = SimpleNamespace(
        execute=lambda tool, arguments: "=== arbitrary future dashboard format ==="
    )
    runner.agent = SimpleNamespace(
        check_day_advanced=lambda output: pytest.fail(
            "dashboard text must not control week advancement"
        )
    )

    assert "arbitrary" in runner._execute_tool("bash", {"command": "next-week"})
