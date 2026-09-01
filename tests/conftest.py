"""Shared fixtures for source-level CEOBench regression tests."""

import pytest
from numpy.random import default_rng

from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.database import init_database
from saas_bench.simulator.simulation import Simulator
from saas_bench.simulator.tools import AgentTools


@pytest.fixture
def make_initialized_sim(tmp_path):
    def _make_initialized_sim(*, config=None, seed=123):
        config = config or BenchmarkConfig(seed=seed)
        conn = init_database(tmp_path / "ceobench.db")
        sim = Simulator(conn, config, default_rng(seed))
        sim.initialize()
        return conn, sim, config

    return _make_initialized_sim


@pytest.fixture
def make_agent_tools(tmp_path):
    def _make_agent_tools(conn, config, *, day=0, seed=123):
        return AgentTools(
            conn,
            current_day=day,
            workspace_path=tmp_path / "workspace",
            rng=default_rng(seed),
            config=config,
            seed=seed,
        )

    return _make_agent_tools
