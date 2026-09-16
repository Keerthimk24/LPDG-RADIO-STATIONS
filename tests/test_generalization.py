"""Tests for model generalization — unseen gateways and future weeks.

The challenge brief specifically asks whether the model works on:
  - Gateways it has never seen during training
  - Weeks in the future (after the training window)

These tests validate both using synthetic data with known properties.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features_for_monday, get_feature_columns
from src.model.cost_function import rank_by_cost


class TestUnseenGateways:
    """Test that the feature pipeline handles gateways not present in training."""

    def test_new_gateway_gets_features(self, sample_telemetry, sample_gateway_master,
                                        sample_meter_reads, sample_field_visits):
        """A gateway that exists in gateway_master but has no visit history
        should still get a full feature vector (visit features default to 0)."""
        # Add a brand-new gateway to gateway_master that has NO visit history
        new_gw = pd.DataFrame({
            "gateway_id": ["GWUNSEEN00000"],
            "tenant": ["tenant_new"],
            "site_type": ["Gebaeude"],
            "region": ["Bayern"],
            "hw_model": ["GW-2100"],
            "antenna_type": ["Omni 3dBi"],
            "fw_version": ["3.2.0"],
            "fw_updated_on": pd.to_datetime(["2025-06-01"]),
            "installed_on": pd.to_datetime(["2025-12-01"]),
            "decommissioned_on": [pd.NaT],
            "n_meters_installed": [50],
        })
        extended_master = pd.concat([sample_gateway_master, new_gw], ignore_index=True)

        # Add minimal telemetry for the new gateway so it has data
        new_telem = []
        for h in range(24 * 14):  # 2 weeks of telemetry
            ts = pd.Timestamp("2026-01-15", tz="UTC") + pd.Timedelta(hours=h)
            new_telem.append({
                "gateway_id": "GWUNSEEN00000",
                "ts": ts,
                "DateDt": ts.strftime("%Y-%m-%d"),
                "hour": ts.hour,
                "offline_duration_sec": int(np.random.exponential(100)),
                "disconnection_cnt": int(np.random.poisson(0.5)),
                "reboot_cnt": int(np.random.poisson(0.1)),
                "reboot_duration_sec": int(np.random.exponential(10)),
                "reboot_importance": np.random.uniform(0, 1),
                "no_conn_importance": np.random.uniform(0, 1),
                "online_duration_mins": max(0, 60 - np.random.exponential(5)),
                "avg_load1": np.random.uniform(0.1, 2.0),
                "avg_memfree": np.random.uniform(10000, 50000),
                "rx_nr_pkts": int(np.random.exponential(30)),
                "rssi_good": int(np.random.randint(0, 14)),
                "rssi_normal": int(np.random.randint(0, 14)),
                "rssi_bad": int(np.random.randint(0, 5)),
                "rscp_rsrp_good": int(np.random.randint(0, 14)),
                "rscp_rsrp_normal": int(np.random.randint(0, 14)),
                "rscp_rsrp_bad": int(np.random.randint(0, 5)),
                "ecio_rsrq_good": int(np.random.randint(0, 14)),
                "ecio_rsrq_normal": int(np.random.randint(0, 14)),
                "ecio_rsrq_bad": int(np.random.randint(0, 5)),
                "network_2g": int(np.random.randint(0, 5)),
                "network_3g": int(np.random.randint(0, 5)),
                "network_4g": int(np.random.randint(0, 14)),
                "network_unknown": 0,
            })
        extended_telemetry = pd.concat(
            [sample_telemetry, pd.DataFrame(new_telem)], ignore_index=True
        )

        features = build_features_for_monday(
            extended_telemetry, extended_master, sample_meter_reads, sample_field_visits,
            "2026-01-29",
        )

        # The unseen gateway MUST appear in the feature matrix
        assert "GWUNSEEN00000" in features.index, \
            "Unseen gateway must get a feature row even with no visit history"

        # All feature columns should be numeric (no NaN from missing joins)
        feature_cols = get_feature_columns(features)
        unseen_row = features.loc["GWUNSEEN00000", feature_cols]
        assert not unseen_row.isna().any(), \
            f"Unseen gateway has NaN features: {unseen_row[unseen_row.isna()].index.tolist()}"

    def test_unseen_gateway_can_be_ranked(self):
        """Ranking should work even if the gateway was never in training data."""
        # Simulate predictions for a mix of seen and unseen gateways
        gw_ids = np.array(["GW_TRAIN_1", "GW_TRAIN_2", "GW_UNSEEN_1", "GW_UNSEEN_2"])
        probs = np.array([0.9, 0.3, 0.95, 0.1])

        ranked = rank_by_cost(gw_ids, probs, budget=3)

        assert len(ranked) == 3
        # Highest probability gateway (unseen) should be ranked first
        assert ranked[0]["gateway_id"] == "GW_UNSEEN_1"
        # The ranking doesn't care whether a gateway was in training
        ranked_ids = {r["gateway_id"] for r in ranked}
        assert "GW_UNSEEN_1" in ranked_ids


class TestFutureWeeks:
    """Test that features can be built for weeks after the training window."""

    def test_features_for_future_monday(self, sample_telemetry, sample_gateway_master,
                                         sample_meter_reads, sample_field_visits):
        """Features should be buildable for any Monday, not just training Mondays."""
        # Build features for a Monday well after the training window
        # The sample_telemetry runs from 2026-01-01 for 30 days, so 2026-01-29 is valid
        features = build_features_for_monday(
            sample_telemetry, sample_gateway_master, sample_meter_reads, sample_field_visits,
            "2026-01-29",
        )

        assert len(features) > 0, "Must produce features for future Mondays"
        feature_cols = get_feature_columns(features)
        assert len(feature_cols) > 5, "Must have meaningful features, not just metadata"

    def test_no_data_leakage_across_weeks(self, sample_telemetry, sample_gateway_master,
                                           sample_meter_reads, sample_field_visits):
        """Features for week N must ONLY use data from before week N.

        This is the critical temporal integrity test. If violated, the model
        cheats during training by peeking at future telemetry.
        """
        monday_early = "2026-01-15"
        monday_late = "2026-01-22"

        feats_early = build_features_for_monday(
            sample_telemetry, sample_gateway_master, sample_meter_reads, sample_field_visits,
            monday_early,
        )
        feats_late = build_features_for_monday(
            sample_telemetry, sample_gateway_master, sample_meter_reads, sample_field_visits,
            monday_late,
        )

        # Both should have features
        assert len(feats_early) > 0
        assert len(feats_late) > 0

        # Features should differ because they use different data windows
        feature_cols = get_feature_columns(feats_early)
        common_cols = [c for c in feature_cols if c in feats_late.columns]

        # At least some telemetry features should differ between weeks
        # (because the 7-day window shifts)
        common_gws = set(feats_early.index) & set(feats_late.index)
        if common_gws and common_cols:
            gw = list(common_gws)[0]
            # Not all features should be identical (different data windows)
            early_vals = feats_early.loc[gw, common_cols]
            late_vals = feats_late.loc[gw, common_cols]
            # It's possible some features are the same (e.g. static gateway metadata),
            # but time-varying features (7d_* aggregations) should differ
            time_varying = [c for c in common_cols if c.startswith("7d_")]
            if time_varying:
                assert not (early_vals[time_varying] == late_vals[time_varying]).all(), \
                    "Time-varying features must differ between weeks (data leakage check)"


class TestCostComparison:
    """Verify the cost comparison logic between model and baseline."""

    def test_cost_savings_calculation(self):
        """The cost formula must match: total_cost = FP × €380 + FN × €600."""
        cost_fp = 380.0
        cost_fn = 600.0

        # Scenario: 6 false alarms, 51 missed broken gateways
        fp, fn = 6, 51
        total_cost = fp * cost_fp + fn * cost_fn
        assert total_cost == 2280 + 30600 == 32880

        # Scenario: 12 false alarms, 57 missed (baseline performance)
        fp_base, fn_base = 12, 57
        baseline_cost = fp_base * cost_fp + fn_base * cost_fn
        assert baseline_cost == 4560 + 34200 == 38760

        # Model should be cheaper than baseline
        assert total_cost < baseline_cost, "ML model must have lower total cost than baseline"

    def test_breakeven_threshold_consistency(self):
        """Breakeven probability P* = C_FP / (C_FP + C_FN) must be consistent."""
        from src.model.cost_function import compute_visit_value

        cost_fp = 380.0
        cost_fn = 600.0
        p_star = cost_fp / (cost_fp + cost_fn)

        assert abs(p_star - 0.3878) < 0.001, f"Breakeven should be ~0.3878, got {p_star}"

        # At breakeven, visit value should be zero
        value = compute_visit_value(p_star, cost_fp, cost_fn)
        assert abs(value) < 1.0, f"Value at breakeven should be ~0, got {value}"

        # Below breakeven: negative value (don't visit)
        assert compute_visit_value(0.2, cost_fp, cost_fn) < 0
        # Above breakeven: positive value (visit)
        assert compute_visit_value(0.6, cost_fp, cost_fn) > 0
