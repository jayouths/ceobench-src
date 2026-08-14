PYTEST := uv run --frozen pytest

.PHONY: test test-fast test-unit test-component test-config test-llm test-simulator test-all test-collect

# 日常默认入口：只执行不依赖外部服务的单元和组件测试。
test: test-fast

test-fast:
	$(PYTEST) tests/unit tests/component -m "not slow and not external"

test-unit:
	$(PYTEST) tests/unit

test-component:
	$(PYTEST) tests/component -m "not slow and not external"

test-config:
	$(PYTEST) tests/unit/saas_bench/test_experiment_config.py

test-llm:
	$(PYTEST) tests/unit/saas_bench/test_llm_provider.py

test-simulator:
	$(PYTEST) tests/component/simulator

# 完整回归只在跨模块改动、合并和正式实验前执行。
test-all:
	$(PYTEST)

test-collect:
	$(PYTEST) --collect-only -q
