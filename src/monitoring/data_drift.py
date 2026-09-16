"""
Data drift monitor — watches incoming data and alerts if it changed shape.

Monitors three types of drift:
1. Schema drift: new/missing columns, type changes
2. Volume drift: missing hours, gateways, time gaps
3. Distribution drift: KS-test on core metrics vs training distribution
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Results of a drift detection run."""
    schema_ok: bool = True
    volume_ok: bool = True
    distribution_ok: bool = True
    alerts: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.schema_ok and self.volume_ok and self.distribution_ok

    def summary(self) -> str:
        status = "OK" if self.is_ok else "DRIFT DETECTED"
        lines = [f"Drift Report: {status}"]
        lines.append(f"  Schema:       {'OK' if self.schema_ok else 'DRIFT'}")
        lines.append(f"  Volume:       {'OK' if self.volume_ok else 'DRIFT'}")
        lines.append(f"  Distribution: {'OK' if self.distribution_ok else 'DRIFT'}")
        for alert in self.alerts:
            lines.append(f"  ALERT: {alert}")
        return "\n".join(lines)


class DataDriftMonitor:
    """Monitors incoming data for drift relative to a reference distribution."""

    # Core metrics to monitor for distribution drift
    MONITORED_METRICS = [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
    ]

    def __init__(self, drift_threshold: float = 0.05):
        """
        Args:
            drift_threshold: KS-test p-value threshold. Below this = drift.
        """
        self.drift_threshold = drift_threshold
        self.reference_stats: dict | None = None

    def fit_reference(self, telemetry: pd.DataFrame) -> dict:
        """Compute reference statistics from training data.

        Args:
            telemetry: Training telemetry DataFrame.

        Returns:
            Reference statistics dict.
        """
        ref_stats = {
            "columns": list(telemetry.columns),
            "n_gateways": telemetry["gateway_id"].nunique(),
            "n_rows": len(telemetry),
            "metrics": {},
        }

        for metric in self.MONITORED_METRICS:
            if metric in telemetry.columns:
                values = telemetry[metric].dropna().values
                ref_stats["metrics"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "sample": values[np.random.choice(len(values), min(10000, len(values)), replace=False)].tolist(),
                }

        self.reference_stats = ref_stats
        logger.info("Reference statistics computed from %d rows, %d gateways",
                     ref_stats["n_rows"], ref_stats["n_gateways"])
        return ref_stats

    def save_reference(self, path: pathlib.Path):
        """Save reference statistics to JSON."""
        if self.reference_stats is None:
            raise ValueError("No reference statistics. Call fit_reference first.")
        with open(path, "w") as f:
            json.dump(self.reference_stats, f, indent=2)
        logger.info("Reference statistics saved to %s", path)

    def load_reference(self, path: pathlib.Path):
        """Load reference statistics from JSON."""
        with open(path) as f:
            self.reference_stats = json.load(f)
        logger.info("Reference statistics loaded from %s", path)

    def check(self, telemetry: pd.DataFrame) -> DriftReport:
        """Run all drift checks on new data.

        Args:
            telemetry: New telemetry DataFrame to check.

        Returns:
            DriftReport with findings.
        """
        if self.reference_stats is None:
            raise ValueError("No reference statistics. Call fit_reference or load_reference first.")

        report = DriftReport()

        self._check_schema(telemetry, report)
        self._check_volume(telemetry, report)
        self._check_distribution(telemetry, report)

        logger.info("Drift check complete:\n%s", report.summary())
        return report

    def _check_schema(self, df: pd.DataFrame, report: DriftReport):
        """Check for schema changes."""
        expected_cols = set(self.reference_stats["columns"])
        actual_cols = set(df.columns)

        missing = expected_cols - actual_cols
        new = actual_cols - expected_cols

        if missing:
            report.schema_ok = False
            report.alerts.append(f"Missing columns: {sorted(missing)}")

        if new:
            report.alerts.append(f"New columns (not in reference): {sorted(new)}")
            # New columns are a warning, not an error
            logger.warning("New columns found: %s", sorted(new))

    def _check_volume(self, df: pd.DataFrame, report: DriftReport):
        """Check for volume anomalies."""
        ref_gateways = self.reference_stats["n_gateways"]
        actual_gateways = df["gateway_id"].nunique()

        # Alert if gateway count changed by >10%
        if abs(actual_gateways - ref_gateways) / ref_gateways > 0.1:
            report.volume_ok = False
            report.alerts.append(
                f"Gateway count changed: {ref_gateways} → {actual_gateways} "
                f"({(actual_gateways - ref_gateways) / ref_gateways:.1%})"
            )

        # Check for time gaps
        if "ts" in df.columns:
            ts_diff = df.sort_values("ts")["ts"].diff()
            max_gap = ts_diff.max()
            if max_gap > pd.Timedelta(hours=48):
                report.alerts.append(f"Time gap of {max_gap} detected (>48h)")

    def _check_distribution(self, df: pd.DataFrame, report: DriftReport):
        """Check for distribution drift using KS-test."""
        for metric in self.MONITORED_METRICS:
            if metric not in df.columns:
                continue
            if metric not in self.reference_stats.get("metrics", {}):
                continue

            ref_sample = np.array(self.reference_stats["metrics"][metric]["sample"])
            new_values = df[metric].dropna().values

            if len(new_values) < 100:
                report.alerts.append(f"Too few values for KS-test on {metric}: {len(new_values)}")
                continue

            # Subsample for efficiency
            if len(new_values) > 10000:
                new_values = np.random.choice(new_values, 10000, replace=False)

            ks_stat, p_value = stats.ks_2samp(ref_sample, new_values)

            if p_value < self.drift_threshold:
                report.distribution_ok = False
                report.alerts.append(
                    f"Distribution drift on {metric}: KS={ks_stat:.4f}, p={p_value:.6f} "
                    f"(threshold={self.drift_threshold})"
                )
                logger.warning("Distribution drift detected on %s (p=%.6f)", metric, p_value)
            else:
                logger.info("No drift on %s (p=%.4f)", metric, p_value)
