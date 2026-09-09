"""按职责拆分的 Harness 回归测试。"""

import json
import random

import sqlite3

from types import SimpleNamespace

import pytest

from saas_bench.simulator.config import BenchmarkConfig

from saas_bench.simulator.customer_llm import (
    CustomerSimulator,
    EVENT_WRITING_ANGLE_POOLS,
    judge_agent_social_post,
)
from saas_bench.simulator.database import (
    add_agent_social_post,
    get_recent_agent_posts_for_judge,
)
from saas_bench.experiment.llm_provider import TextLLMResult

from saas_bench.runtime.server_entry import (
    _apply_simulator_llm_config,
    _apply_simulator_llm_env_overrides,
    _restore_simulator_llm_config,
)


from tests.support.harness import (
    RecordingOpenAI,
)

def test_simulator_settings_survive_environment_and_session_round_trip(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("LOCAL_LLM_KEY", "test-key")
    overrides = {
        "social_post_llm_api_type": "openai_responses",
        "social_post_llm_model": "social-test",
        "social_post_llm_base_url": "http://social.test/v1",
        "social_post_llm_api_key_env": "LOCAL_LLM_KEY",
        "social_post_llm_reasoning_effort": "none",
        "social_post_llm_temperature": 0.31,
        "social_post_llm_top_p": 0.72,
        "social_post_llm_max_tokens": 123,
        "social_post_llm_timeout_seconds": 45.0,
        "social_post_llm_response_cache_path": str(tmp_path / "social-cache"),
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


def test_identical_social_request_reuses_shared_response_cache(tmp_path):
    social = RecordingOpenAI()
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(
            social_post_llm_api_type="openai_responses",
            social_post_llm_model="social-test",
            social_post_llm_max_tokens=100,
            social_post_llm_response_cache_path=str(tmp_path / "social-cache"),
        ),
        social_client=social,
    )

    first = simulator.create_social_response("system", "user", task="customer_post")
    second = simulator.create_social_response("system", "user", task="customer_post")

    assert len(social.responses.calls) == 1
    assert first.response_cache_hit is False
    assert second.response_cache_hit is True
    assert second.text == first.text
    assert (second.input_tokens, second.cached_tokens, second.output_tokens) == (0, 0, 0)


def test_customer_post_prompt_only_depends_on_semantic_event_identity():
    config = BenchmarkConfig(
        seed=42,
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_max_tokens=100,
    )
    prefetched = {
        "persona": {"name": "Test Customer"},
        "group_chars": {"description": "Test group", "social_media_tone": "direct"},
        "product_name": "NovaMind",
        "company_name": "NovaMind AI",
    }
    event_context = {"event_type": "overload", "penalty": 0.2}
    clients = [RecordingOpenAI(), RecordingOpenAI()]

    # 即使进程全局 random 处于不同位置，同一经营事件也必须形成相同请求。
    random_state = random.getstate()
    try:
        for global_seed, client in zip((0, 1), clients):
            random.seed(global_seed)
            simulator = CustomerSimulator(
                conn=sqlite3.connect(":memory:"),
                config=config,
                social_client=client,
            )
            simulator.generate_social_post(
                day=7,
                customer_id=101,
                satisfaction=0.4,
                group_id="S1",
                sentiment="negative",
                post_type="perceived_quality_penalty",
                event_context=event_context,
                _prefetched=prefetched,
                _skip_log_cost=True,
            )
    finally:
        random.setstate(random_state)

    assert clients[0].responses.calls == clients[1].responses.calls


@pytest.mark.parametrize(
    "event_type",
    ["overload", "outage", "issue", "quota", "contract_dissatisfaction"],
)
def test_quality_event_post_only_uses_compatible_writing_angle(event_type):
    social = RecordingOpenAI()
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(
            seed=42,
            social_post_llm_api_type="openai_responses",
            social_post_llm_model="social-test",
            social_post_llm_max_tokens=100,
        ),
        social_client=social,
    )

    simulator.generate_social_post(
        day=7,
        customer_id=101,
        satisfaction=0.4,
        group_id="S1",
        sentiment="negative",
        post_type="perceived_quality_penalty",
        event_context={"event_type": event_type, "penalty": 0.2},
        _prefetched={"product_name": "NovaMind", "company_name": "NovaMind AI"},
        _skip_log_cost=True,
    )

    system_prompt = social.responses.calls[0]["input"][0]["content"]
    angle_line = next(
        line for line in system_prompt.splitlines() if line.startswith("Writing Angle: ")
    )
    assert angle_line.removeprefix("Writing Angle: ") in EVENT_WRITING_ANGLE_POOLS[
        event_type
    ]

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
        "response_cache_hit": False,
    })]

def test_local_model_cost_is_zero_when_explicitly_configured():
    config = BenchmarkConfig(
        social_post_llm_model="DeepSeek-V4-Flash",
        social_post_llm_pricing={"DeepSeek-V4-Flash": {
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
        1_000_000, 1_000_000, 0, purpose="customer_social_post"
    )
    assert cost.amount == 0.0
    assert cost.currency == "CNY"

def test_unknown_model_cost_requires_explicit_pricing():
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(social_post_llm_model="unknown-model"),
    )

    with pytest.raises(ValueError, match="No token pricing configured"):
        simulator._calculate_cost(1, 1, 0, purpose="customer_social_post")


def test_agent_post_judge_only_treats_earlier_posts_as_history(
    make_initialized_sim,
):
    conn, _, config = make_initialized_sim(seed=42)
    content = "We reduced API latency by 40%."
    prompts = []

    def capture_prompt(system_prompt, user_prompt, task):
        prompts.append((system_prompt, user_prompt, task))
        return TextLLMResult(
            text="SCORE: 0.5\nREASON: Useful update.",
            model="social-test",
            input_tokens=1,
            output_tokens=1,
            cached_tokens=0,
            reasoning_tokens=None,
            raw_response=None,
        )

    first_post_id = add_agent_social_post(conn, day=0, content=content)
    first_history = get_recent_agent_posts_for_judge(conn, 0, first_post_id)
    judge_agent_social_post(
        capture_prompt,
        config,
        content,
        "S1",
        "small developers",
        "direct",
        10,
        100.0,
        first_history,
    )

    assert first_history == []
    assert "Their recent posts:" not in prompts[-1][1]
    assert prompts[-1][1].count(content) == 1

    second_post_id = add_agent_social_post(conn, day=7, content=content)
    second_history = get_recent_agent_posts_for_judge(conn, 7, second_post_id)
    judge_agent_social_post(
        capture_prompt,
        config,
        content,
        "S1",
        "small developers",
        "direct",
        10,
        100.0,
        second_history,
    )

    assert [post["content"] for post in second_history] == [content]
    assert "Their recent posts:" in prompts[-1][1]
    assert prompts[-1][1].count(content) == 2
