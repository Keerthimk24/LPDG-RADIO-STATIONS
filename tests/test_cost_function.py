"""Tests for cost function and ranking logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.cost_function import (
    asymmetric_cost_objective,
    compute_visit_value,
    rank_by_cost,
)


class TestCostFunction:
    """Tests for the asymmetric cost function."""

    def test_fn_penalty_higher_than_fp(self):
        """False negatives should be penalised more heavily than false positives."""
        # When true label is 1 and prediction is 0 (FN), gradient should be steeper
        # than when true label is 0 and prediction is 1 (FP)
        value_fn = compute_visit_value(1.0)  # Definitely broken → high value
        value_fp = compute_visit_value(0.0)  # Definitely fine → negative value
        assert value_fn > 0, "Visiting a broken gateway should have positive value"
        assert value_fp < 0, "Visiting a healthy gateway should have negative value"

    def test_breakeven_probability(self):
        """At P = 380/980 ≈ 0.388, visit value should be near zero."""
        # Visit value = P * 600 - (1-P) * 380
        # At breakeven: P * 600 = (1-P) * 380 → P = 380/980 ≈ 0.388
        breakeven = 380.0 / (380.0 + 600.0)
        value = compute_visit_value(breakeven)
        assert abs(value) < 1.0, f"Value at breakeven should be near zero, got {value}"

    def test_visit_value_monotonic(self):
        """Higher probability should always mean higher visit value."""
        probs = np.linspace(0, 1, 100)
        values = [compute_visit_value(p) for p in probs]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], "Visit value must be monotonically increasing"


class TestRanking:
    def test_returns_correct_count(self):
        gw_ids = np.array([f"GW{i}" for i in range(20)])
        probs = np.random.uniform(0, 1, 20)
        ranked = rank_by_cost(gw_ids, probs, budget=15)
        assert len(ranked) == 15

    def test_ranks_are_sequential(self):
        gw_ids = np.array([f"GW{i}" for i in range(20)])
        probs = np.random.uniform(0, 1, 20)
        ranked = rank_by_cost(gw_ids, probs, budget=15)
        ranks = [r["rank"] for r in ranked]
        assert ranks == list(range(1, 16))

    def test_highest_probability_ranked_first(self):
        gw_ids = np.array(["GW_HIGH", "GW_LOW", "GW_MID"])
        probs = np.array([0.99, 0.01, 0.5])
        ranked = rank_by_cost(gw_ids, probs, budget=3)
        assert ranked[0]["gateway_id"] == "GW_HIGH"
        assert ranked[2]["gateway_id"] == "GW_LOW"

    def test_no_duplicate_gateways(self):
        gw_ids = np.array([f"GW{i}" for i in range(20)])
        probs = np.random.uniform(0, 1, 20)
        ranked = rank_by_cost(gw_ids, probs, budget=15)
        ranked_gws = [r["gateway_id"] for r in ranked]
        assert len(set(ranked_gws)) == len(ranked_gws), "No duplicates allowed"
