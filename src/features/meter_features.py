"""
Meter read success features — measures how well each gateway is performing
its primary job (relaying meter readings).

Declining meter read success is a lagging but high-confidence signal of gateway trouble.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_meter_features(
    meter_reads: pd.DataFrame,
    monday: str | dt.date,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Build meter read features for a single prediction Monday.

    Args:
        meter_reads: meter_read_success DataFrame with columns:
            gateway_id, week_start, meters_expected, meters_read, read_success_rate
        monday: The prediction Monday.
        lookback_weeks: Number of weeks to look back for averaging.

    Returns:
        DataFrame indexed by gateway_id with meter read features.
    """
    if isinstance(monday, str):
        monday = dt.date.fromisoformat(monday)

    monday_ts = pd.Timestamp(monday)

    # Only use data strictly before this Monday
    historical = meter_reads[meter_reads["week_start"] < monday_ts].copy()

    if historical.empty:
        logger.warning("No meter read data before %s", monday)
        return pd.DataFrame()

    features = {}

    # Latest week's data (most recent week before the prediction Monday)
    latest_week = historical.groupby("gateway_id").apply(
        lambda g: g.sort_values("week_start").iloc[-1]
    )

    features["meter_read_rate_latest"] = latest_week["read_success_rate"]
    features["meters_expected"] = latest_week["meters_expected"]
    features["meters_read_latest"] = latest_week["meters_read"]
    features["meters_at_risk"] = (
        latest_week["meters_expected"] * (1 - latest_week["read_success_rate"])
    )

    # Rolling average over lookback weeks
    cutoff = monday_ts - pd.Timedelta(weeks=lookback_weeks)
    recent = historical[historical["week_start"] >= cutoff]

    if not recent.empty:
        avg_by_gw = recent.groupby("gateway_id")["read_success_rate"].mean()
        features["meter_read_rate_avg"] = avg_by_gw

        # Trend: latest vs average (negative = declining)
        features["meter_read_rate_delta"] = (
            features["meter_read_rate_latest"] - avg_by_gw
        )

        # Volatility: std of read success rate
        std_by_gw = recent.groupby("gateway_id")["read_success_rate"].std().fillna(0)
        features["meter_read_rate_std"] = std_by_gw

        # Worst week in lookback period
        min_by_gw = recent.groupby("gateway_id")["read_success_rate"].min()
        features["meter_read_rate_min"] = min_by_gw

        # Number of weeks with data
        features["meter_weeks_available"] = recent.groupby("gateway_id").size()

        # Weeks with read rate below 80%
        poor_weeks = recent[recent["read_success_rate"] < 0.8]
        features["meter_poor_weeks"] = poor_weeks.groupby("gateway_id").size().reindex(
            avg_by_gw.index, fill_value=0
        )
    else:
        features["meter_read_rate_avg"] = features["meter_read_rate_latest"]
        features["meter_read_rate_delta"] = 0.0
        features["meter_read_rate_std"] = 0.0
        features["meter_read_rate_min"] = features["meter_read_rate_latest"]
        features["meter_weeks_available"] = 0
        features["meter_poor_weeks"] = 0

    result = pd.DataFrame(features)
    result.index.name = "gateway_id"
    result = result.fillna(0)

    logger.info("Built meter read features: %d gateways, %d features",
                len(result), len(result.columns))
    return result
