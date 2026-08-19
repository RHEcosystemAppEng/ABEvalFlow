"""Tests for abevalflow.observability.cost — cost estimation from token counts."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from abevalflow.observability.cost import _load_model_costs, _resolve_rates, calculate_cost


@pytest.fixture()
def costs_file(tmp_path: Path) -> Path:
    p = tmp_path / "model_costs.yaml"
    p.write_text(
        dedent("""\
        claude-sonnet:
          input_per_1k: 0.003
          output_per_1k: 0.015
        gpt-4o:
          input_per_1k: 0.005
          output_per_1k: 0.015
        _default:
          input_per_1k: 0.002
          output_per_1k: 0.010
        """)
    )
    return p


class TestLoadModelCosts:
    def test_load_valid_file(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        costs = _load_model_costs(costs_file)
        assert "claude-sonnet" in costs
        assert costs["claude-sonnet"]["input_per_1k"] == 0.003

    def test_load_missing_file(self, tmp_path: Path) -> None:
        _load_model_costs.cache_clear()
        costs = _load_model_costs(tmp_path / "nonexistent.yaml")
        assert costs == {}

    def test_load_empty_file(self, tmp_path: Path) -> None:
        _load_model_costs.cache_clear()
        p = tmp_path / "empty.yaml"
        p.write_text("")
        costs = _load_model_costs(p)
        assert costs == {}


class TestResolveRates:
    def test_exact_match(self) -> None:
        costs = {"claude-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015}}
        assert _resolve_rates("claude-sonnet", costs) == costs["claude-sonnet"]

    def test_prefix_match(self) -> None:
        costs = {"claude-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015}}
        assert _resolve_rates("claude-sonnet-4-20250514", costs) == costs["claude-sonnet"]

    def test_default_fallback(self) -> None:
        costs = {
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "_default": {"input_per_1k": 0.002, "output_per_1k": 0.010},
        }
        assert _resolve_rates("unknown-model", costs) == costs["_default"]

    def test_no_match_no_default(self) -> None:
        costs = {"gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015}}
        assert _resolve_rates("unknown-model", costs) == {}


class TestCalculateCost:
    def test_basic_calculation(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(1000, 500, "claude-sonnet", costs_path=costs_file)
        # (1000/1000)*0.003 + (500/1000)*0.015 = 0.003 + 0.0075 = 0.0105
        assert cost == 0.0105

    def test_zero_tokens(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(0, 0, "claude-sonnet", costs_path=costs_file)
        assert cost == 0.0

    def test_no_model_name(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(1000, 500, None, costs_path=costs_file)
        assert cost is None

    def test_unknown_model_uses_default(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(1000, 500, "some-unknown-model", costs_path=costs_file)
        # (1000/1000)*0.002 + (500/1000)*0.010 = 0.002 + 0.005 = 0.007
        assert cost == 0.007

    def test_missing_costs_file(self, tmp_path: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(1000, 500, "claude-sonnet", costs_path=tmp_path / "nope.yaml")
        assert cost is None

    def test_prefix_match_versioned_model(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(2000, 1000, "claude-sonnet-4-20250514", costs_path=costs_file)
        # (2000/1000)*0.003 + (1000/1000)*0.015 = 0.006 + 0.015 = 0.021
        assert cost == 0.021

    def test_precision(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(15000, 3500, "claude-sonnet", costs_path=costs_file)
        # (15000/1000)*0.003 + (3500/1000)*0.015 = 0.045 + 0.0525 = 0.0975
        assert cost == 0.0975

    def test_large_token_counts(self, costs_file: Path) -> None:
        _load_model_costs.cache_clear()
        cost = calculate_cost(1_000_000, 500_000, "claude-sonnet", costs_path=costs_file)
        # (1M/1000)*0.003 + (500K/1000)*0.015 = 3.0 + 7.5 = 10.5
        assert cost == 10.5
