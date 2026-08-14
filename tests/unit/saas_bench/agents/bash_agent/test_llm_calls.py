"""Bash Agent 对应逻辑的快速单元测试。"""

import json

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, Message

from saas_bench.agents.bash_agent import run_test

from saas_bench.agents.bash_agent.run_test import BashAgentRunner


def test_decision_response_cost_uses_the_served_model(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.model = "requested"
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
    runner.response_log_file = tmp_path / "responses.jsonl"
    runner.agent = SimpleNamespace(
        last_input_tokens=1_000_000,
        last_output_tokens=1_000_000,
        last_cached_tokens=250_000,
        last_reasoning_tokens=125_000,
        last_serving_model="served",
    )

    runner._log_response(1, 0, [], {"model": "served"})

    entry = json.loads(runner.response_log_file.read_text())
    assert entry["served_model"] == "served"
    assert entry["pricing_model"] == "official"
    assert entry["cached_tokens"] == 250_000
    assert entry["reasoning_tokens"] == 125_000
    assert entry["cost_amount"] == pytest.approx(6.3125)
    assert entry["currency"] == "CNY"
    assert runner.total_decision_agent_cost_by_currency == {
        "CNY": pytest.approx(6.3125)
    }

def test_decision_agent_request_builder_uses_config_without_hidden_defaults():
    agent = BashAgent.__new__(BashAgent)
    agent.model = "decision-test"
    agent.max_output_tokens = 345
    agent.temperature = 0.51
    agent.top_p = 0.92
    agent.reasoning_effort = "none"
    agent.request_options = {}
    agent._get_system_prompt_with_memory = lambda: "system"

    params = agent._build_openai_responses_kwargs([{"role": "user", "content": "x"}], [])

    assert params["max_output_tokens"] == 345
    assert params["temperature"] == pytest.approx(0.51)
    assert params["top_p"] == pytest.approx(0.92)
    assert params["reasoning"] == {"effort": "none", "summary": "auto"}

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

def test_anthropic_agent_does_not_retry_local_errors():
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = []
    agent._get_system_prompt_with_memory = lambda: (
        _ for _ in ()
    ).throw(OSError("local failure"))

    with pytest.raises(OSError, match="local failure"):
        agent._call_anthropic()

def test_anthropic_agent_does_not_create_unpriced_prompt_cache():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="served-anthropic",
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tool-1",
                    name="bash",
                    input={"command": "pwd"},
                )
            ],
        )

    agent = BashAgent(
        tool_descriptions=[],
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test-model",
        api_type="anthropic_messages",
        max_invalid_responses_per_turn=2,
        max_output_tokens=100,
    )
    # 模拟旧 checkpoint 遗留的缓存断点，新请求不应继续携带它。
    agent.conversation = [
        Message(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "dashboard",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    action = agent._call_anthropic()

    assert action == run_test.Action(tool="bash", arguments={"command": "pwd"})
    assert len(calls) == 1
    assert "cache_control" not in json.dumps(calls[0])
    assert agent.last_input_tokens == 11
    assert agent.last_output_tokens == 7

def test_anthropic_agent_does_not_retry_bad_requests():
    import anthropic
    import httpx

    error = anthropic.BadRequestError(
        "bad request",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://example.test"),
        ),
        body={},
    )
    messages = SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(error)
    )
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = []
    agent._get_system_prompt_with_memory = lambda: "system"
    agent.model = "test-model"
    agent.max_output_tokens = 100
    agent.temperature = None
    agent.top_p = None
    agent.request_options = {}
    agent.client = SimpleNamespace(messages=messages)

    with pytest.raises(anthropic.BadRequestError):
        agent._call_anthropic()
