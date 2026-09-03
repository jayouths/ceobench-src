from types import SimpleNamespace

import pytest

from saas_bench.experiment.llm_provider import (
    API_OPENAI_CHAT,
    API_OPENAI_RESPONSES,
    MissingModelPricingError,
    api_tool_choice,
    call_text_model,
    create_llm_client,
    model_token_cost,
    openai_chat_request_parameters,
    openai_chat_cached_tokens,
    token_cost,
)


@pytest.mark.parametrize(
    ("api_type", "policy", "expected"),
    [
        (API_OPENAI_CHAT, "required", "required"),
        (API_OPENAI_RESPONSES, "required", "required"),
        (API_OPENAI_CHAT, "auto", "auto"),
    ],
)
def test_tool_choice_is_forwarded_to_openai_protocols(api_type, policy, expected):
    assert api_tool_choice(api_type, policy) == expected


def test_unknown_api_type_is_rejected_before_client_creation():
    with pytest.raises(ValueError, match="api_type must be one of"):
        create_llm_client(
            api_type="unknown",
            api_key="test",
            base_url=None,
            timeout_seconds=10,
        )


class Recorder:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_openai_responses_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        output_text="  response text  ",
        model="served-responses",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=7),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    ))
    client = SimpleNamespace(responses=recorder)

    result = call_text_model(
        client=client,
        api_type=API_OPENAI_RESPONSES,
        model="requested",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=100,
        temperature=0.4,
        top_p=None,
        reasoning_effort=None,
        request_options={"extra_body": {"enable_thinking": False}},
    )

    assert result.text == "response text"
    assert (result.model, result.input_tokens, result.output_tokens) == (
        "served-responses", 12, 5,
    )
    assert result.cached_tokens == 7
    assert result.reasoning_tokens == 3
    assert recorder.calls == [{
        "model": "requested",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "max_output_tokens": 100,
        "temperature": 0.4,
        "extra_body": {"enable_thinking": False},
    }]


def test_openai_chat_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        model="served-chat",
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="chat text"))],
    ))
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))

    result = call_text_model(
        client=client,
        api_type=API_OPENAI_CHAT,
        model="requested",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=50,
        temperature=None,
        top_p=0.8,
        reasoning_effort="high",
    )

    assert (result.text, result.model, result.input_tokens, result.output_tokens) == (
        "chat text", "served-chat", 8, 3,
    )
    assert result.cached_tokens == 2
    assert result.reasoning_tokens == 1
    assert recorder.calls[0]["reasoning_effort"] == "high"
    assert recorder.calls[0]["max_tokens"] == 50
    assert recorder.calls[0]["top_p"] == pytest.approx(0.8)


def test_responses_rejects_missing_served_model():
    client = SimpleNamespace(responses=Recorder(SimpleNamespace(
        output_text="text",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )))

    with pytest.raises(ValueError, match="non-empty model identifier"):
        call_text_model(
            client=client,
            api_type=API_OPENAI_RESPONSES,
            model="requested",
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=10,
            temperature=None,
            top_p=None,
            reasoning_effort=None,
        )


def test_chat_completions_rejects_empty_served_model():
    response = SimpleNamespace(
        model="  ",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Recorder(response)))

    with pytest.raises(ValueError, match="non-empty model identifier"):
        call_text_model(
            client=client,
            api_type=API_OPENAI_CHAT,
            model="requested",
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=10,
            temperature=None,
            top_p=None,
            reasoning_effort=None,
        )


def test_deepseek_chat_cache_hit_tokens_are_normalized():
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        prompt_cache_hit_tokens=9,
        prompt_cache_miss_tokens=3,
    )
    recorder = Recorder(SimpleNamespace(
        model="deepseek-v4-pro",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
    ))

    result = call_text_model(
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
        api_type=API_OPENAI_CHAT,
        model="deepseek-v4-pro",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=50,
        temperature=None,
        top_p=None,
        reasoning_effort="high",
    )

    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.cached_tokens == 9
    assert result.reasoning_tokens is None


def test_explicit_zero_reasoning_tokens_is_preserved():
    recorder = Recorder(SimpleNamespace(
        model="served-chat",
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=3,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
    ))

    result = call_text_model(
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
        api_type=API_OPENAI_CHAT,
        model="requested",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=50,
        temperature=None,
        top_p=None,
        reasoning_effort=None,
    )

    assert result.reasoning_tokens == 0


def test_chat_cache_fields_must_not_disagree():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        prompt_cache_hit_tokens=3,
    )

    with pytest.raises(ValueError, match="Conflicting cached-token counts"):
        openai_chat_cached_tokens(usage)


def test_chat_parameters_and_private_options_are_forwarded():
    params = openai_chat_request_parameters(
        model="channel-model",
        max_output_tokens=4096,
        temperature=None,
        top_p=None,
        reasoning_effort="low",
        tool_choice="required",
        request_options={
            "extra_body": {
                "do_sample": False,
                "stream": False,
                "thinking": {"type": "enabled", "clear_thinking": True},
                "response_format": {"type": "text"},
            }
        },
    )

    assert params["max_tokens"] == 4096
    assert "max_completion_tokens" not in params
    assert params["reasoning_effort"] == "low"
    assert params["tool_choice"] == "required"
    assert params["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": True,
    }


def test_cost_uses_served_model_and_rejects_unknown_model():
    pricing = {
        "requested": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.1,
            "output_cost_per_million": 2.0,
        },
        "served": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 3.0,
            "cached_input_cost_per_million": 0.25,
            "output_cost_per_million": 4.0,
        },
    }

    cost = model_token_cost("served", 1_000_000, 1_000_000, 250_000, pricing)
    assert cost.amount == pytest.approx(6.3125)
    assert cost.currency == "CNY"
    assert cost.pricing_model == "served"
    with pytest.raises(MissingModelPricingError, match="unlisted"):
        model_token_cost("unlisted", 1, 1, 0, pricing)


def test_cost_maps_channel_model_to_canonical_pricing_model():
    pricing = {
        "deepseek-v4-pro": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.32,
            "cached_input_cost_per_million": 0.044,
            "output_cost_per_million": 3.96,
        }
    }

    cost = model_token_cost(
        "deepseek-v4-pro-0813",
        1_000_000,
        1_000_000,
        250_000,
        pricing,
        {"deepseek-v4-pro-0813": "deepseek-v4-pro"},
    )

    assert cost.amount == pytest.approx(4.961)
    assert cost.currency == "USD"
    assert cost.pricing_model == "deepseek-v4-pro"


@pytest.mark.parametrize(
    ("cached_tokens", "expected"),
    [(0, 3.0), (250_000, 2.525), (1_000_000, 1.1)],
)
def test_cost_splits_cached_and_uncached_input(cached_tokens, expected):
    assert token_cost(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_tokens=cached_tokens,
        uncached_input_cost_per_million=2.0,
        cached_input_cost_per_million=0.1,
        output_cost_per_million=10.0,
    ) == pytest.approx(expected)


def test_cost_rejects_cached_tokens_above_total_input():
    with pytest.raises(ValueError, match="cannot exceed"):
        token_cost(10, 1, 11, 1.0, 0.1, 2.0)
