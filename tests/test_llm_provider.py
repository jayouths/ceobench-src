from types import SimpleNamespace

import pytest

from saas_bench.llm_provider import (
    API_ANTHROPIC_MESSAGES,
    API_OPENAI_CHAT,
    API_OPENAI_RESPONSES,
    MissingModelPricingError,
    call_text_model,
    model_token_cost_usd,
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
        usage=SimpleNamespace(input_tokens=12, output_tokens=5),
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
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
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
    assert recorder.calls[0]["reasoning_effort"] == "high"
    assert recorder.calls[0]["max_completion_tokens"] == 50
    assert recorder.calls[0]["top_p"] == pytest.approx(0.8)


def test_anthropic_messages_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        model="served-anthropic",
        content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        usage=SimpleNamespace(input_tokens=20, output_tokens=9),
    ))
    client = SimpleNamespace(messages=recorder)

    result = call_text_model(
        client=client,
        api_type=API_ANTHROPIC_MESSAGES,
        model="claude-sonnet-test",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=200,
        temperature=None,
        top_p=None,
        reasoning_effort=None,
        request_options={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
    )

    assert (result.text, result.model, result.input_tokens, result.output_tokens) == (
        "first\nsecond", "served-anthropic", 20, 9,
    )
    assert recorder.calls[0]["thinking"] == {"type": "adaptive"}
    assert recorder.calls[0]["output_config"] == {"effort": "medium"}


def test_cost_uses_served_model_and_rejects_unknown_model():
    pricing = {
        "requested": {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0},
        "served": {"input_cost_per_million": 3.0, "output_cost_per_million": 4.0},
    }

    assert model_token_cost_usd("served", 1_000_000, 1_000_000, pricing) == 7.0
    with pytest.raises(MissingModelPricingError, match="unlisted"):
        model_token_cost_usd("unlisted", 1, 1, pricing)
