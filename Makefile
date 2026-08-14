PYTEST := uv run --frozen pytest

.PHONY: test test-fast test-unit test-component test-config test-llm test-agent test-checkpoint test-resume test-api test-simulator test-integration test-all test-collect

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

test-agent:
	$(PYTEST) tests/unit/saas_bench/agents/bash_agent tests/component/bash_agent -m "not slow and not external"

test-checkpoint:
	$(PYTEST) tests/integration/bash_agent/test_checkpoint.py

test-resume:
	$(PYTEST) tests/integration/bash_agent/test_resume.py

test-api:
	$(PYTEST) tests/integration/api

test-simulator:
	$(PYTEST) tests/component/simulator

test-integration:
	$(PYTEST) tests/integration -m "not slow and not external"

# 完整回归只在跨模块改动、合并和正式实验前执行。
test-all:
	$(PYTEST)

test-collect:
	$(PYTEST) --collect-only -q
