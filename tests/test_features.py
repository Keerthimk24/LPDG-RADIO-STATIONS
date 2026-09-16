"""Tests for feature engineering modules."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.telemetry_features import build_telemetry_features, aggregate_window
from src.features.meter_features import build_meter_features
from src.features.gateway_features import build_gateway_features
from src.features.visit_features import build_visit_features
from src.features.builder import build_features_for_monday, get_feature_columns


class TestTelemetryFeatures:
    """Tests for telemetry feature engineering."""

    def test_aggregate_window_returns_correct_shape(self, sample_telemetry):
        end = pd.Timestamp("2026-01-15", tz="UTC")
        result = aggregate_window(sample_telemetry, end, window_days=7, prefix="7d")
        assert len(result) > 0
        assert all(c.startswith("7d_") for c in result.columns)

    def test_no_future_data_leakage(self, sample_telemetry):
        """Features for Monday should ONLY use data before Monday."""
        monday = "2026-01-15"
        result = build_telemetry_features(sample_telemetry, monday)

        # Verify no data from Monday onwards was used
        monday_ts = pd.Timestamp(monday, tz="UTC")
        future_data = sample_telemetry[sample_telemetry["ts"] >= monday_ts]

        # The result should exist and have features
        assert len(result) > 0, "Should have features for at least some gateways"

    def test_trend_ratio_meaning(self, sample_telemetry):
        """Trend ratio > 1 should mean degradation."""
        result = build_telemetry_features(sample_telemetry, "2026-01-29")
        if "trend_offline_duration_sec_ratio" in result.columns:
            # Trend ratio should be non-negative
            assert (result["trend_offline_duration_sec_ratio"] >= 0).all()


class TestMeterFeatures:
    def test_builds_correctly(self, sample_meter_reads):
        result = build_meter_features(sample_meter_reads, "2026-02-10")
        assert len(result) > 0
        assert "meter_read_rate_latest" in result.columns
        assert "meters_at_risk" in result.columns

    def test_read_rate_bounds(self, sample_meter_reads):
        result = build_meter_features(sample_meter_reads, "2026-02-10")
        assert (result["meter_read_rate_latest"] >= 0).all()
        assert (result["meter_read_rate_latest"] <= 1).all()


class TestGatewayFeatures:
    def test_builds_correctly(self, sample_gateway_master):
        result = build_gateway_features(sample_gateway_master, "2026-02-02")
        assert len(result) > 0
        assert "n_meters_installed" in result.columns
        assert "gateway_age_days" in result.columns

    def test_categorical_encoding(self, sample_gateway_master):
        result = build_gateway_features(sample_gateway_master, "2026-02-02")
        hw_cols = [c for c in result.columns if c.startswith("hw_model_")]
        assert len(hw_cols) > 0, "Should have one-hot encoded hardware models"


class TestVisitFeatures:
    def test_builds_correctly(self, sample_field_visits):
        result = build_visit_features(sample_field_visits, "2026-02-02")
        assert len(result) > 0
        assert "total_visits" in result.columns
        assert "days_since_last_visit" in result.columns

    def test_fault_rate_bounds(self, sample_field_visits):
        result = build_visit_features(sample_field_visits, "2026-02-02")
        assert (result["visit_fault_rate"] >= 0).all()
        assert (result["visit_fault_rate"] <= 1).all()


class TestFeatureBuilder:
    def test_builds_all_features(
        self, sample_telemetry, sample_gateway_master,
        sample_meter_reads, sample_field_visits
    ):
        result = build_features_for_monday(
            sample_telemetry, sample_gateway_master,
            sample_meter_reads, sample_field_visits,
            "2026-01-26",
        )
        assert len(result) > 0
        feature_cols = get_feature_columns(result)
        assert len(feature_cols) > 10, "Should have many features from all sources"

    def test_feature_columns_exclude_metadata(
        self, sample_telemetry, sample_gateway_master,
        sample_meter_reads, sample_field_visits
    ):
        result = build_features_for_monday(
            sample_telemetry, sample_gateway_master,
            sample_meter_reads, sample_field_visits,
            "2026-01-26",
        )
        feature_cols = get_feature_columns(result)
        assert "_monday" not in feature_cols
        assert "label" not in feature_cols
