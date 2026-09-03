"""Bash Agent 对应逻辑的快速单元测试。"""

import json

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent

from saas_bench.agents.bash_agent.runner import BashAgentRunner


def test_reasoning_token_total_becomes_unknown_after_unreported_call():
    agent = BashAgent.__new__(BashAgent)
    agent.total_reasoning_tokens = 0

    agent._record_reasoning_tokens(0)
    assert agent.last_reasoning_tokens == 0
    assert agent.total_reasoning_tokens == 0

    agent._record_reasoning_tokens(None)
    assert agent.last_reasoning_tokens is None
    assert agent.total_reasoning_tokens is None

    # 累计值一旦不完整，后续已上报调用也不能让它重新变成伪完整总量。
    agent._record_reasoning_tokens(5)
    assert agent.last_reasoning_tokens == 5
    assert agent.total_reasoning_tokens is None


def test_decision_response_cost_uses_the_served_model(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.model = "requested"
    runner.api_type = "openai_responses"
    runner.pricing = {
        "official": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 3.0,
            "cached_input_cost_per_million": 0.25,
            "output_cost_per_million": 4.0,
        },
    }
    runner.pricing_model_map = {
        "requested": "official",
        "served": "official",
    }
    runner.total_decision_agent_cost_by_currency = {}
    runner.run_id = "test"
    runner.trajectory_log_file = tmp_path / "trajectory.jsonl"
    runner.performance_log_file = tmp_path / "performance.jsonl"
    runner._experiment_log_writer = None
    runner._pending_decision_context = None
    runner.agent = SimpleNamespace(
        last_input_tokens=1_000_000,
        last_output_tokens=1_000_000,
        last_cached_tokens=250_000,
        last_reasoning_tokens=125_000,
        last_serving_model="served",
    )

    runner._log_decision_llm_call(1, 0, [], {"model": "served"}, 1.25)

    entry = json.loads(runner.trajectory_log_file.read_text())
    assert entry["event_type"] == "llm_call"
    assert entry["react_round"] == 1
    assert entry["elapsed_seconds"] == pytest.approx(1.25)
    assert entry["served_model"] == "served"
    assert entry["pricing_model"] == "official"
    assert entry["cached_tokens"] == 250_000
    assert entry["reasoning_tokens"] == 125_000
    assert entry["cost_amount"] == pytest.approx(6.3125)
    assert entry["currency"] == "CNY"
    assert runner.total_decision_agent_cost_by_currency == {
        "CNY": pytest.approx(6.3125)
    }


def _make_response_logging_runner(tmp_path, initial_observation, analysis_enabled):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.model = "requested"
    runner.api_type = "openai_responses"
    runner.pricing = {
        "official": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.0,
            "output_cost_per_million": 1.0,
        },
    }
    runner.pricing_model_map = {"served": "official"}
    runner.total_decision_agent_cost_by_currency = {}
    runner.run_id = "test"
    runner.workspace_dir = tmp_path
    runner.trajectory_log_file = tmp_path / "trajectory.jsonl"
    runner.performance_log_file = tmp_path / "performance.jsonl"
    runner._experiment_log_writer = None
    runner.analysis_enabled = analysis_enabled
    runner.agent = SimpleNamespace(
        last_input_tokens=10,
        last_output_tokens=5,
        last_cached_tokens=0,
        last_reasoning_tokens=0,
        last_serving_model="served",
        initial_observation_for_audit=initial_observation,
    )
    return runner


def test_analysis_initial_observation_is_auditable_without_repeated_context(tmp_path):
    brief = "# STRATEGY BRIEF\n\n经营状态"
    observation = f"dashboard\n\n---\n\n{brief}"
    brief_path = tmp_path / "analysis/day_007/STRATEGY_BRIEF.md"
    brief_path.parent.mkdir(parents=True)
    brief_path.write_text(brief)
    runner = _make_response_logging_runner(tmp_path, observation, True)
    runner._pending_decision_context = {
        "dashboard": "dashboard",
        "strategy_brief": brief,
        "strategy_brief_artifact": "analysis/day_007/STRATEGY_BRIEF.md",
        "decision_observation": observation,
    }

    # Chat Completions 的 messages 含 system + user；审计值取自 Agent
    # 实际追加的 observation，因此不依赖具体 Provider 的消息封装。
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": observation},
    ]
    raw_response = {
        "output": [
            {"type": "function_call", "name": "bash", "arguments": "{}"}
        ]
    }
    runner._log_decision_llm_call(1, 7, messages, raw_response)
    runner.agent.initial_observation_for_audit = None
    runner._log_decision_llm_call(2, 7, messages, raw_response)

    observation_event, first_call, second_call = [
        json.loads(line)
        for line in runner.trajectory_log_file.read_text().splitlines()
    ]
    assert observation_event["event_type"] == "decision_observation"
    assert observation_event["dashboard"] == "dashboard"
    assert observation_event["strategy_brief"] == brief
    assert observation_event["rendered_observation"] == observation
    assert first_call["event_type"] == "llm_call"
    assert second_call["event_type"] == "llm_call"


def test_baseline_initial_observation_records_original_dashboard(tmp_path):
    runner = _make_response_logging_runner(tmp_path, "baseline dashboard", False)
    runner._pending_decision_context = {
        "dashboard": "baseline dashboard",
        "strategy_brief": None,
        "strategy_brief_artifact": None,
        "decision_observation": "baseline dashboard",
    }

    runner._log_decision_llm_call(
        1,
        0,
        [{"role": "user", "content": "baseline dashboard"}],
        {"output": [{"type": "function_call", "name": "bash", "arguments": "{}"}]},
    )

    observation_event, call_event = [
        json.loads(line)
        for line in runner.trajectory_log_file.read_text().splitlines()
    ]
    assert observation_event["dashboard"] == "baseline dashboard"
    assert observation_event["strategy_brief"] is None
    assert observation_event["rendered_observation"] == "baseline dashboard"
    assert call_event["status"] == "valid"


def test_response_callback_consumes_initial_observation_once():
    agent = BashAgent.__new__(BashAgent)
    agent.total_turns = 1
    agent.current_day = 7
    agent._initial_observation_for_audit = "weekly observation"
    captured = []
    agent.response_callback = lambda **kwargs: captured.append(
        (agent.initial_observation_for_audit, kwargs["elapsed_seconds"])
    )

    agent._emit_response_callback([], {"id": "first"}, 1.25)
    agent._emit_response_callback([], {"id": "retry"}, 0.75)

    assert captured == [("weekly observation", 1.25), (None, 0.75)]
    assert agent.initial_observation_for_audit is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "bash", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
            ("valid", 1, None),
        ),
        (
            {
                "output": [
                    {"type": "function_call", "name": "bash", "arguments": "{"}
                ]
            },
            ("invalid", 1, "invalid_tool_arguments"),
        ),
        ({"content": [{"type": "text", "text": "answer"}]},
         ("invalid", 0, "missing_tool_call")),
    ],
)
def test_decision_response_status_matches_harness_tool_validation(
    response, expected
):
    assert BashAgentRunner._decision_response_status(response) == expected

def test_decision_agent_request_builder_uses_config_without_hidden_defaults():
    agent = BashAgent.__new__(BashAgent)
    agent.model = "decision-test"
    agent.api_type = "openai_responses"
    agent.max_output_tokens = 345
    agent.temperature = 0.51
    agent.top_p = 0.92
    agent.reasoning_effort = "none"
    agent.tool_choice = "required"
    agent.request_options = {}
    agent._get_system_prompt_with_memory = lambda: "system"

    params = agent._build_openai_responses_kwargs([{"role": "user", "content": "x"}], [])

    assert params["max_output_tokens"] == 345
    assert params["temperature"] == pytest.approx(0.51)
    assert params["top_p"] == pytest.approx(0.92)
    assert params["reasoning"] == {"effort": "none", "summary": "auto"}
    assert params["tool_choice"] == "required"

    agent.temperature = None
    agent.top_p = None
    agent.reasoning_effort = None
    omitted = agent._build_openai_responses_kwargs([], [])
    assert "temperature" not in omitted
    assert "top_p" not in omitted
    assert "reasoning" not in omitted

@pytest.mark.parametrize(
    ("call_method", "builder_name"),
    [
        ("_call_openai", "_build_openai_chat_kwargs"),
        ("_call_openai_responses", "_build_openai_responses_kwargs"),
    ],
)
def test_openai_agent_does_not_retry_local_errors(call_method, builder_name):
    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    setattr(
        agent,
        builder_name,
        lambda *args: (_ for _ in ()).throw(OSError("local failure")),
    )

    with pytest.raises(OSError, match="local failure"):
        getattr(agent, call_method)()

@pytest.mark.parametrize(
    ("call_method", "builder_name", "client"),
    [
        (
            "_call_openai",
            "_build_openai_chat_kwargs",
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        ),
        (
            "_call_openai_responses",
            "_build_openai_responses_kwargs",
            SimpleNamespace(responses=SimpleNamespace()),
        ),
    ],
)
def test_openai_agent_does_not_retry_bad_requests(call_method, builder_name, client):
    import httpx
    import openai

    error = openai.BadRequestError(
        "bad request",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://example.test"),
        ),
        body={},
    )
    endpoint = (
        client.chat.completions
        if call_method == "_call_openai"
        else client.responses
    )
    endpoint.create = lambda **kwargs: (_ for _ in ()).throw(error)

    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = client
    setattr(agent, builder_name, lambda *args: {})

    with pytest.raises(openai.BadRequestError):
        getattr(agent, call_method)()

@pytest.mark.parametrize(
    ("call_method", "builder_name", "client"),
    [
        (
            "_call_openai",
            "_build_openai_chat_kwargs",
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        ),
        (
            "_call_openai_responses",
            "_build_openai_responses_kwargs",
            SimpleNamespace(responses=SimpleNamespace()),
        ),
    ],
)
def test_openai_agent_does_not_add_unbounded_provider_retries(
    call_method, builder_name, client
):
    import httpx
    import openai

    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise openai.APIConnectionError(
            request=httpx.Request("POST", "http://example.test")
        )

    endpoint = (
        client.chat.completions
        if call_method == "_call_openai"
        else client.responses
    )
    endpoint.create = fail
    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = client
    setattr(agent, builder_name, lambda *args: {})

    with pytest.raises(openai.APIConnectionError):
        getattr(agent, call_method)()

    assert calls == 1

def test_openai_responses_stops_after_configured_invalid_response_limit():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="test-model", usage=None, output=[])

    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent.model = "test-model"
    agent.max_invalid_responses_per_turn = 2
    agent.total_turns = 0
    agent._consecutive_errors = 0
    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.total_cached_tokens = 0
    agent.total_reasoning_tokens = 0
    agent.response_callback = None
    agent.tool_result_callback = None
    agent._build_openai_responses_kwargs = lambda *args: {}

    with pytest.raises(RuntimeError, match="2 responses"):
        agent._call_openai_responses()

    assert len(calls) == 2
    assert agent.total_reasoning_tokens is None
