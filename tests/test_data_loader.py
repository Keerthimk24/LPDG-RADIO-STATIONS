"""Tests for data loading and validation."""

from __future__ import annotations

import pytest

from src.data.loader import normalise_gateway_id
from src.data.validator import (
    validate_telemetry,
    validate_gateway_master,
    validate_meter_reads,
    ValidationResult,
)


class TestGatewayIdNormalisation:
    """Gateway IDs must be consistent across all data sources."""

    def test_bare_hex(self):
        assert normalise_gateway_id("0639EA5602C1") == "0639EA5602C1"

    def test_colon_format(self):
        assert normalise_gateway_id("06:39:EA:56:02:C1") == "0639EA5602C1"

    def test_lowercase(self):
        assert normalise_gateway_id("0639ea5602c1") == "0639EA5602C1"

    def test_mixed_case_colons(self):
        assert normalise_gateway_id("06:39:ea:56:02:C1") == "0639EA5602C1"


class TestTelemetryValidation:
    def test_valid_telemetry_passes(self, sample_telemetry):
        result = validate_telemetry(sample_telemetry)
        assert result.passed

    def test_missing_column_fails(self, sample_telemetry):
        broken = sample_telemetry.drop(columns=["gateway_id"])
        result = validate_telemetry(broken)
        assert not result.passed

    def test_invalid_hour_detected(self, sample_telemetry):
        broken = sample_telemetry.copy()
        broken.loc[0, "hour"] = 25  # Invalid hour
        result = validate_telemetry(broken)
        assert any("hour" in e for e in result.errors)


class TestGatewayMasterValidation:
    def test_valid_master_passes(self, sample_gateway_master):
        result = validate_gateway_master(sample_gateway_master)
        assert result.passed

    def test_duplicate_ids_detected(self, sample_gateway_master):
        import pandas as pd
        broken = pd.concat([sample_gateway_master, sample_gateway_master.iloc[:1]])
        result = validate_gateway_master(broken)
        assert not result.passed


class TestMeterReadValidation:
    def test_valid_meter_reads_passes(self, sample_meter_reads):
        result = validate_meter_reads(sample_meter_reads)
        assert result.passed


class TestValidationResult:
    def test_default_passes(self):
        r = ValidationResult()
        assert r.passed

    def test_error_fails(self):
        r = ValidationResult()
        r.add_error("test error")
        assert not r.passed
        assert len(r.errors) == 1

    def test_warning_still_passes(self):
        r = ValidationResult()
        r.add_warning("test warning")
        assert r.passed
        assert len(r.warnings) == 1
