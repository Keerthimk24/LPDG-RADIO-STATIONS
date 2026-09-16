"""Tests for data drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_drift import DataDriftMonitor, DriftReport


class TestDataDriftMonitor:
    """Tests for the drift detection system."""

    def test_no_drift_on_same_data(self, sample_telemetry):
        """Same data should show no drift."""
        monitor = DataDriftMonitor(drift_threshold=0.05)
        monitor.fit_reference(sample_telemetry)
        report = monitor.check(sample_telemetry)
        assert report.distribution_ok, f"Should be no drift on same data: {report.summary()}"

    def test_detects_schema_drift(self, sample_telemetry):
        """Removing a column should trigger schema drift."""
        monitor = DataDriftMonitor()
        monitor.fit_reference(sample_telemetry)

        # Drop a critical column
        modified = sample_telemetry.drop(columns=["offline_duration_sec"])
        report = monitor.check(modified)
        assert not report.schema_ok, "Should detect missing column"

    def test_detects_distribution_drift(self, sample_telemetry):
        """Dramatically different data should trigger distribution drift."""
        monitor = DataDriftMonitor(drift_threshold=0.05)
        monitor.fit_reference(sample_telemetry)

        # Create wildly different distribution
        modified = sample_telemetry.copy()
        modified["offline_duration_sec"] = np.random.exponential(5000, len(modified))
        modified["disconnection_cnt"] = np.random.poisson(50, len(modified))
        modified["reboot_cnt"] = np.random.poisson(10, len(modified))

        report = monitor.check(modified)
        assert not report.distribution_ok, "Should detect distribution shift"

    def test_save_load_reference(self, sample_telemetry, tmp_path):
        """Reference stats should survive save/load cycle."""
        monitor = DataDriftMonitor()
        monitor.fit_reference(sample_telemetry)

        ref_path = tmp_path / "reference.json"
        monitor.save_reference(ref_path)

        monitor2 = DataDriftMonitor()
        monitor2.load_reference(ref_path)

        report = monitor2.check(sample_telemetry)
        assert report.distribution_ok


class TestDriftReport:
    def test_ok_by_default(self):
        report = DriftReport()
        assert report.is_ok

    def test_not_ok_with_schema_drift(self):
        report = DriftReport(schema_ok=False)
        assert not report.is_ok

    def test_summary_contains_alerts(self):
        report = DriftReport()
        report.alerts.append("test alert")
        assert "test alert" in report.summary()
