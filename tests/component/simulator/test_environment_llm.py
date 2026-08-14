"""按职责拆分的 Harness 回归测试。"""

import json

import sqlite3

from types import SimpleNamespace

import pytest

from saas_bench.config import BenchmarkConfig

from saas_bench.customer_llm import CustomerSimulator

from saas_bench.server_entry import (
    _apply_simulator_llm_config,
    _apply_simulator_llm_env_overrides,
    _restore_simulator_llm_config,
)


from tests.support.harness import (
    RecordingOpenAI,
)

def test_simulator_settings_survive_environment_and_session_round_trip(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_KEY", "test-key")
    overrides = {
        "social_post_llm_provider": "openai",
        "social_post_llm_api_type": "openai_responses",
        "social_post_llm_model": "social-test",
        "social_post_llm_base_url": "http://social.test/v1",
        "social_post_llm_api_key_env": "LOCAL_LLM_KEY",
        "social_post_llm_reasoning_effort": "none",
        "social_post_llm_temperature": 0.31,
        "social_post_llm_top_p": 0.72,
        "social_post_llm_max_tokens": 123,
        "social_post_llm_timeout_seconds": 45.0,
        "social_post_llm_pricing": {"official-social": {
            "currency": "USD",
            "uncached_input_cost_per_million": 0.0,
            "cached_input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
        }},
        "social_post_llm_pricing_model_map": {
            "social-test": "official-social",
        },
    }
    monkeypatch.setenv("CEOBENCH_SIMULATOR_LLM_CONFIG", json.dumps(overrides))

    created = BenchmarkConfig()
    _apply_simulator_llm_env_overrides(created)
    session_values = _apply_simulator_llm_config(created)
    restored = BenchmarkConfig()
    _restore_simulator_llm_config(restored, {"simulator_llm": session_values})

    for field, expected in overrides.items():
        assert getattr(restored, field) == expected

def test_social_openai_calls_receive_configured_api_parameters():
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_reasoning_effort="none",
        social_post_llm_temperature=0.31,
        social_post_llm_top_p=0.72,
        social_post_llm_max_tokens=123,
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
        social_client=social,
    )

    simulator.create_social_response("social system", "social user")

    assert social.responses.calls == [{
        "model": "social-test",
        "input": [
            {"role": "system", "content": "social system"},
            {"role": "user", "content": "social user"},
        ],
        "max_output_tokens": 123,
        "temperature": 0.31,
        "top_p": 0.72,
        "reasoning": {"effort": "none"},
    }]

def test_simulator_text_model_rejects_empty_success_response():
    client = RecordingOpenAI()
    client.responses.create = lambda **kwargs: SimpleNamespace(
        output_text="",
        model="social-test",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(
            social_post_llm_provider="openai_compatible",
            social_post_llm_api_type="openai_responses",
            social_post_llm_model="social-test",
            social_post_llm_max_tokens=100,
        ),
        social_client=client,
    )

    with pytest.raises(RuntimeError, match="returned an empty response"):
        simulator.create_social_response("system", "user", task="customer_post")

def test_task_request_options_merge_without_dropping_model_options():
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai_compatible",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_max_tokens=100,
        social_post_llm_request_options={
            "extra_body": {"enable_thinking": False, "priority": "normal"},
            "extra_headers": {"x-base": "base"},
        },
        social_post_llm_task_parameters={
            "customer_post": {
                "max_output_tokens": 50,
                "request_options": {
                    "extra_body": {"priority": "high"},
                    "extra_query": {"trace": "1"},
                },
            }
        },
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
        social_client=social,
    )

    simulator.create_social_response("system", "user", task="customer_post")

    assert social.responses.calls[0]["max_output_tokens"] == 50
    assert social.responses.calls[0]["extra_body"] == {
        "enable_thinking": False,
        "priority": "high",
    }
    assert social.responses.calls[0]["extra_headers"] == {"x-base": "base"}
    assert social.responses.calls[0]["extra_query"] == {"trace": "1"}

def test_customer_social_post_preserves_served_model_and_logs_cost(
    make_initialized_sim,
):
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_max_tokens=100,
        social_post_llm_pricing={
            "social-test": {
                "currency": "CNY",
                "uncached_input_cost_per_million": 1.0,
                "cached_input_cost_per_million": 0.1,
                "output_cost_per_million": 2.0,
            }
        },
    )
    conn, simulation, _ = make_initialized_sim(config=config)
    customer_simulator = CustomerSimulator(
        conn=conn,
        config=config,
        social_client=social,
    )
    customer_simulator.simulator = simulation
    simulation.customer_simulator = customer_simulator
    simulation.current_day = 7

    customer = conn.execute(
        "SELECT customer_id, group_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()
    async_state = simulation._submit_social_posts_async(
        regular_work=[{
            "customer_id": customer["customer_id"],
            "group_id": customer["group_id"],
            "satisfaction": 0.5,
            "is_churned": False,
            "post_type": "general_satisfaction",
            "event_context": None,
        }],
        influence_cache={},
        macro_work=[],
    )
    simulation._collect_social_posts_async(async_state)

    post = conn.execute(
        "SELECT content FROM social_media_posts ORDER BY post_id DESC LIMIT 1"
    ).fetchone()
    cost = conn.execute(
        """
        SELECT model, purpose, input_tokens, cached_tokens, output_tokens,
               cost_amount, currency
        FROM api_costs ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    assert post["content"] == "response"
    assert dict(cost) == {
        "model": "social-test",
        "purpose": "customer_social_post",
        "input_tokens": 11,
        "cached_tokens": 0,
        "output_tokens": 7,
        "cost_amount": pytest.approx(0.000025),
        "currency": "CNY",
    }

def test_successful_zero_token_social_call_is_still_logged(make_initialized_sim):
    conn, simulation, _config = make_initialized_sim()
    customer_id = conn.execute(
        "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()[0]
    customer_simulator = SimpleNamespace(
        _log_cost=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    simulation.customer_simulator = customer_simulator
    simulation.current_day = 7
    calls = []
    simulation._process_social_post_results(
        [{
            "type": "macro",
            "customer_id": customer_id,
            "pmi": 50.0,
            "trend": "flat",
            "text": "macro post",
            "success": True,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "model": "local-model",
        }],
        {},
    )

    assert calls == [((7, "macro_social_post", 0, 0), {
        "cached_tokens": 0,
        "model": "local-model",
    })]

def test_local_model_cost_is_zero_when_explicitly_configured():
    config = BenchmarkConfig(
        social_post_llm_pricing={"qwen3-coder:30b": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 0.0,
            "cached_input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
        }},
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
    )

    cost = simulator._calculate_cost(
        1_000_000, 1_000_000, 0,
        model="qwen3-coder:30b", purpose="customer_social_post"
    )
    assert cost.amount == 0.0
    assert cost.currency == "CNY"

def test_unknown_model_cost_requires_explicit_pricing():
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(),
    )

    with pytest.raises(ValueError, match="No token pricing configured"):
        simulator._calculate_cost(
            1, 1, 0, model="unknown-model", purpose="customer_social_post"
        )
