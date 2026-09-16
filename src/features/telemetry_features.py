"""
Telemetry feature engineering — aggregates hourly telemetry into per-gateway features
for a given prediction Monday.

All features use ONLY data strictly before the prediction Monday (no data leakage).
Two windows: trailing 7 days (recent) and trailing 28 days (baseline).
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Core health metrics for aggregation
CORE_METRICS = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt"]

# Extended metrics
EXTENDED_METRICS = [
    "reboot_importance", "no_conn_importance",
    "online_duration_mins", "avg_load1", "avg_memfree",
    "reboot_duration_sec", "rx_nr_pkts",
]

# Signal quality columns
SIGNAL_COLS_BAD = ["rssi_bad", "rscp_rsrp_bad", "ecio_rsrq_bad"]
SIGNAL_COLS_GOOD = ["rssi_good", "rscp_rsrp_good", "ecio_rsrq_good"]
SIGNAL_COLS_NORMAL = ["rssi_normal", "rscp_rsrp_normal", "ecio_rsrq_normal"]

# Network type columns
NETWORK_COLS = ["network_2g", "network_3g", "network_4g", "network_unknown"]


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, returning default if b is zero or NaN."""
    if b == 0 or pd.isna(b) or pd.isna(a):
        return default
    return a / b


def aggregate_window(
    telemetry: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate telemetry for each gateway over a trailing window.

    Args:
        telemetry: Full telemetry DataFrame with 'ts' and 'gateway_id'.
        end_date: Exclusive upper bound (typically a Monday at midnight UTC).
        window_days: Number of days to look back.
        prefix: Column name prefix (e.g., '7d' or '28d').

    Returns:
        DataFrame indexed by gateway_id with aggregated features.
    """
    start = end_date - dt.timedelta(days=window_days)
    window = telemetry[(telemetry["ts"] >= start) & (telemetry["ts"] < end_date)]

    if window.empty:
        logger.warning("Empty window for %s (%s to %s)", prefix, start, end_date)
        return pd.DataFrame()

    grouped = window.groupby("gateway_id")

    # Core metric aggregations
    aggs = {}
    for metric in CORE_METRICS:
        col = grouped[metric]
        aggs[f"{prefix}_{metric}_mean"] = col.mean()
        aggs[f"{prefix}_{metric}_max"] = col.max()
        aggs[f"{prefix}_{metric}_sum"] = col.sum()
        aggs[f"{prefix}_{metric}_std"] = col.std().fillna(0)
        aggs[f"{prefix}_{metric}_nonzero_hours"] = col.apply(lambda s: (s > 0).sum())

    # Extended metric aggregations (lighter — just mean and max)
    for metric in EXTENDED_METRICS:
        if metric in window.columns:
            col = grouped[metric]
            aggs[f"{prefix}_{metric}_mean"] = col.mean()
            aggs[f"{prefix}_{metric}_max"] = col.max()

    # Signal quality: ratio of bad readings
    for bad_col, good_col, norm_col in zip(SIGNAL_COLS_BAD, SIGNAL_COLS_GOOD, SIGNAL_COLS_NORMAL):
        if all(c in window.columns for c in [bad_col, good_col, norm_col]):
            total = grouped[[bad_col, good_col, norm_col]].sum()
            total_sum = total.sum(axis=1).replace(0, np.nan)
            aggs[f"{prefix}_{bad_col}_ratio"] = (total[bad_col] / total_sum).fillna(0)

    # Network type: dominant network
    for net_col in NETWORK_COLS:
        if net_col in window.columns:
            aggs[f"{prefix}_{net_col}_ratio"] = (
                grouped[net_col].sum() / grouped[NETWORK_COLS].sum().sum(axis=1).replace(0, np.nan)
            ).fillna(0)

    # Hours of data available (coverage)
    aggs[f"{prefix}_data_hours"] = grouped.size()

    result = pd.DataFrame(aggs)
    result.index.name = "gateway_id"
    return result


def compute_trend_features(
    features_7d: pd.DataFrame,
    features_28d: pd.DataFrame,
) -> pd.DataFrame:
    """Compute trend features: 7d metric / 28d metric ratio.

    Values > 1 mean the metric is worsening (recent is worse than baseline).
    """
    trends = pd.DataFrame(index=features_7d.index)

    for metric in CORE_METRICS:
        mean_7d = features_7d.get(f"7d_{metric}_mean", pd.Series(dtype=float))
        mean_28d = features_28d.get(f"28d_{metric}_mean", pd.Series(dtype=float))

        # Trend ratio: > 1 means degradation
        ratio = mean_7d / mean_28d.replace(0, np.nan)
        trends[f"trend_{metric}_ratio"] = ratio.fillna(1.0).clip(0, 100)

        # Absolute change
        trends[f"trend_{metric}_delta"] = (mean_7d - mean_28d.reindex(mean_7d.index, fill_value=0)).fillna(0)

    # Data coverage trend
    hours_7d = features_7d.get("7d_data_hours", pd.Series(dtype=float))
    hours_28d = features_28d.get("28d_data_hours", pd.Series(dtype=float))
    expected_7d = 7 * 24  # 168 hours
    expected_28d = 28 * 24  # 672 hours
    trends["coverage_7d"] = (hours_7d / expected_7d).fillna(0).clip(0, 1)
    trends["coverage_28d"] = (hours_28d / expected_28d).fillna(0).clip(0, 1)

    trends.index.name = "gateway_id"
    return trends


def build_telemetry_features(
    telemetry: pd.DataFrame,
    monday: str | dt.date,
    baseline_days: int = 28,
    recent_days: int = 7,
) -> pd.DataFrame:
    """Build all telemetry features for a single prediction Monday.

    Args:
        telemetry: Full telemetry DataFrame.
        monday: The prediction Monday (YYYY-MM-DD or date object).
        baseline_days: Window for baseline statistics (default 28).
        recent_days: Window for recent behavior (default 7).

    Returns:
        DataFrame with one row per gateway, telemetry features as columns.
    """
    if isinstance(monday, str):
        monday = dt.date.fromisoformat(monday)
    end_ts = pd.Timestamp(monday, tz="UTC")

    logger.info("Building telemetry features for Monday %s", monday)

    # Aggregate recent and baseline windows
    features_7d = aggregate_window(telemetry, end_ts, recent_days, prefix="7d")
    features_28d = aggregate_window(telemetry, end_ts, baseline_days, prefix="28d")

    if features_7d.empty:
        logger.warning("No 7-day data for Monday %s", monday)
        return pd.DataFrame()

    # Compute trend features
    trends = compute_trend_features(features_7d, features_28d)

    # Join all features
    result = features_7d.join(features_28d, how="left").join(trends, how="left")
    result = result.fillna(0)

    logger.info("Built telemetry features: %d gateways, %d features", len(result), len(result.columns))
    return result
