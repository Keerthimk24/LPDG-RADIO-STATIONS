"""
Data validator — schema checks, quality checks, and drift-relevant validation.

Stops and complains when the data is wrong, instead of quietly giving a wrong answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation run."""
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning("VALIDATION WARNING: %s", msg)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False
        logger.error("VALIDATION ERROR: %s", msg)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Validation {status}: {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


# Expected schema definitions
TELEMETRY_REQUIRED_COLS = [
    "gateway_id", "ts", "DateDt", "hour",
    "offline_duration_sec", "disconnection_cnt", "reboot_cnt",
    "reboot_importance", "no_conn_importance",
    "online_duration_mins", "avg_load1", "avg_memfree",
    "rx_nr_pkts",
]

GATEWAY_MASTER_REQUIRED_COLS = [
    "gateway_id", "tenant", "site_type", "region",
    "hw_model", "antenna_type", "fw_version",
    "installed_on", "n_meters_installed",
]

METER_READ_REQUIRED_COLS = [
    "gateway_id", "week_start", "meters_expected", "meters_read", "read_success_rate",
]


def validate_telemetry(df: pd.DataFrame) -> ValidationResult:
    """Validate telemetry DataFrame schema and quality."""
    result = ValidationResult()

    # Schema check
    missing = [c for c in TELEMETRY_REQUIRED_COLS if c not in df.columns]
    if missing:
        result.add_error(f"Telemetry missing columns: {missing}")
        return result

    # Null checks on critical columns
    for col in ["gateway_id", "ts", "offline_duration_sec", "disconnection_cnt", "reboot_cnt"]:
        null_pct = df[col].isna().mean() * 100
        if null_pct > 5:
            result.add_warning(f"Telemetry column '{col}' has {null_pct:.1f}% nulls")

    # Range checks
    if (df["hour"] < 0).any() or (df["hour"] > 23).any():
        result.add_error("Telemetry 'hour' has values outside 0-23")

    if (df["offline_duration_sec"] < 0).any():
        result.add_warning("Telemetry 'offline_duration_sec' has negative values")

    # Check for duplicate (gateway_id, ts) pairs
    dup_count = df.duplicated(subset=["gateway_id", "ts"]).sum()
    if dup_count > 0:
        result.add_warning(f"Telemetry has {dup_count} duplicate (gateway_id, ts) pairs")

    # Time range check
    ts_range = (df["ts"].max() - df["ts"].min()).days
    if ts_range < 180:
        result.add_warning(f"Telemetry covers only {ts_range} days (expected ~240)")

    n_gateways = df["gateway_id"].nunique()
    if n_gateways < 200:
        result.add_warning(f"Telemetry has only {n_gateways} gateways (expected ~320)")

    logger.info("Telemetry validation: %s", "PASS" if result.passed else "FAIL")
    return result


def validate_gateway_master(df: pd.DataFrame) -> ValidationResult:
    """Validate gateway master DataFrame."""
    result = ValidationResult()

    missing = [c for c in GATEWAY_MASTER_REQUIRED_COLS if c not in df.columns]
    if missing:
        result.add_error(f"Gateway master missing columns: {missing}")
        return result

    # Check for duplicate gateway IDs
    dup_count = df["gateway_id"].duplicated().sum()
    if dup_count > 0:
        result.add_error(f"Gateway master has {dup_count} duplicate gateway_ids")

    # Check n_meters_installed range
    if (df["n_meters_installed"] <= 0).any():
        result.add_warning("Some gateways have 0 or negative meters installed")

    logger.info("Gateway master validation: %s", "PASS" if result.passed else "FAIL")
    return result


def validate_meter_reads(df: pd.DataFrame) -> ValidationResult:
    """Validate meter read success DataFrame."""
    result = ValidationResult()

    missing = [c for c in METER_READ_REQUIRED_COLS if c not in df.columns]
    if missing:
        result.add_error(f"Meter read success missing columns: {missing}")
        return result

    # Check read success rate bounds
    if (df["read_success_rate"] < 0).any() or (df["read_success_rate"] > 1).any():
        result.add_warning("read_success_rate has values outside [0, 1]")

    # Check for weeks where meters_read > meters_expected
    overcounts = (df["meters_read"] > df["meters_expected"]).sum()
    if overcounts > 0:
        result.add_warning(f"{overcounts} rows where meters_read > meters_expected")

    logger.info("Meter read validation: %s", "PASS" if result.passed else "FAIL")
    return result


def validate_all(datasets: dict[str, pd.DataFrame]) -> ValidationResult:
    """Run all validations and return combined result."""
    combined = ValidationResult()

    checks = [
        ("telemetry", validate_telemetry),
        ("gateway_master", validate_gateway_master),
        ("meter_read_success", validate_meter_reads),
    ]

    for name, validator in checks:
        if name in datasets:
            result = validator(datasets[name])
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            if not result.passed:
                combined.passed = False
        else:
            combined.add_warning(f"Dataset '{name}' not found in datasets dict")

    logger.info("Overall validation: %s\n%s", "PASS" if combined.passed else "FAIL", combined.summary())
    return combined
