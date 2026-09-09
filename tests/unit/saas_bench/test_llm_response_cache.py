"""环境 LLM 内容缓存的确定性和并发回归测试。"""

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from saas_bench.experiment.llm_provider import TextLLMResult
from saas_bench.experiment.llm_response_cache import LLMResponseCache


def _result(text: str = "stable response") -> TextLLMResult:
    return TextLLMResult(
        text=text,
        model="served-model",
        input_tokens=11,
        output_tokens=7,
        cached_tokens=2,
        reasoning_tokens=None,
        raw_response={"id": "provider-response"},
    )


def _request_cache_from_process(cache_dir, call_log, barrier, results):
    cache = LLMResponseCache(cache_dir)

    def call_provider():
        with open(call_log, "a", encoding="utf-8") as file:
            file.write("provider-called\n")
            file.flush()
        time.sleep(0.1)
        return _result()

    barrier.wait()
    response = cache.get_or_create({"user_prompt": "shared"}, call_provider)
    results.put(response.response_cache_hit)


def test_identical_request_reuses_response_without_provider_cost(tmp_path):
    cache = LLMResponseCache(tmp_path / "cache")
    provider_calls = 0

    def call_provider():
        nonlocal provider_calls
        provider_calls += 1
        return _result()

    request = {"model": "test", "system_prompt": "s", "user_prompt": "u"}
    first = cache.get_or_create(request, call_provider)
    second = cache.get_or_create(request, call_provider)

    assert provider_calls == 1
    assert first.text == second.text == "stable response"
    assert first.response_cache_hit is False
    assert second.response_cache_hit is True
    assert (second.input_tokens, second.cached_tokens, second.output_tokens) == (0, 0, 0)


def test_different_complete_requests_do_not_share_response(tmp_path):
    cache = LLMResponseCache(tmp_path / "cache")
    provider_calls = 0

    def call_provider():
        nonlocal provider_calls
        provider_calls += 1
        return _result(f"response-{provider_calls}")

    first = cache.get_or_create({"user_prompt": "first"}, call_provider)
    second = cache.get_or_create({"user_prompt": "second"}, call_provider)

    assert provider_calls == 2
    assert first.text == "response-1"
    assert second.text == "response-2"


def test_concurrent_identical_requests_only_call_provider_once(tmp_path):
    cache = LLMResponseCache(tmp_path / "cache")
    provider_calls = 0
    counter_lock = threading.Lock()

    def call_provider():
        nonlocal provider_calls
        with counter_lock:
            provider_calls += 1
        time.sleep(0.05)
        return _result()

    request = {"model": "test", "system_prompt": "s", "user_prompt": "u"}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: cache.get_or_create(request, call_provider),
                range(4),
            )
        )

    assert provider_calls == 1
    assert [result.response_cache_hit for result in results].count(False) == 1
    assert [result.response_cache_hit for result in results].count(True) == 3


def test_cross_process_identical_requests_only_call_provider_once(tmp_path):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    cache_dir = tmp_path / "cache"
    call_log = tmp_path / "provider-calls.log"
    processes = [
        context.Process(
            target=_request_cache_from_process,
            args=(cache_dir, call_log, barrier, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    assert call_log.read_text().splitlines() == ["provider-called"]
    assert sorted(results.get(timeout=1) for _ in range(2)) == [False, True]
