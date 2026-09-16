"""
Feature builder — orchestrates all feature engineering pipelines into a single
feature matrix for a given prediction Monday.

This is the single entry point for feature construction, used by both
training and prediction pipelines.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import pandas as pd

from src.features.telemetry_features import build_telemetry_features
from src.features.meter_features import build_meter_features
from src.features.gateway_features import build_gateway_features
from src.features.visit_features import build_visit_features

logger = logging.getLogger(__name__)


def build_features_for_monday(
    telemetry: pd.DataFrame,
    gateway_master: pd.DataFrame,
    meter_reads: pd.DataFrame,
    field_visits: pd.DataFrame,
    monday: str | dt.date,
    baseline_days: int = 28,
    recent_days: int = 7,
) -> pd.DataFrame:
    """Build the complete feature matrix for a single prediction Monday.

    Joins telemetry, meter read, gateway static, and visit history features
    into one row per gateway.

    Args:
        telemetry: Full telemetry DataFrame.
        gateway_master: Gateway asset register.
        meter_reads: Meter read success rates.
        field_visits: Historical field visits.
        monday: The prediction Monday.
        baseline_days: Telemetry baseline window.
        recent_days: Telemetry recent window.

    Returns:
        DataFrame indexed by gateway_id with all features.
    """
    if isinstance(monday, str):
        monday = dt.date.fromisoformat(monday)

    logger.info("=" * 60)
    logger.info("Building features for Monday %s", monday)

    # 1. Telemetry features (the biggest feature group)
    telem_feats = build_telemetry_features(
        telemetry, monday,
        baseline_days=baseline_days,
        recent_days=recent_days,
    )

    # 2. Meter read features
    meter_feats = build_meter_features(meter_reads, monday)

    # 3. Gateway static features
    gateway_feats = build_gateway_features(gateway_master, monday)

    # 4. Visit history features
    visit_feats = build_visit_features(field_visits, monday)

    # Start with gateway features (ensures we have all active gateways)
    if gateway_feats.empty:
        logger.error("No gateway features for Monday %s", monday)
        return pd.DataFrame()

    result = gateway_feats.copy()

    # Left-join other feature groups
    if not telem_feats.empty:
        result = result.join(telem_feats, how="left")

    if not meter_feats.empty:
        result = result.join(meter_feats, how="left")

    if not visit_feats.empty:
        result = result.join(visit_feats, how="left")

    # Fill NaN from missing joins (gateway exists but no telemetry/visits etc.)
    result = result.fillna(0)

    # Add the prediction date as metadata (not a feature)
    result["_monday"] = monday.isoformat()

    logger.info("Feature matrix for %s: %d gateways × %d features",
                monday, len(result), len(result.columns) - 1)
    return result


def build_features_for_all_weeks(
    telemetry: pd.DataFrame,
    gateway_master: pd.DataFrame,
    meter_reads: pd.DataFrame,
    field_visits: pd.DataFrame,
    mondays: list[str],
    baseline_days: int = 28,
    recent_days: int = 7,
) -> pd.DataFrame:
    """Build features for multiple prediction Mondays and stack them.

    Returns a DataFrame with (gateway_id, _monday) identifying each row.
    """
    all_features = []
    for monday in mondays:
        features = build_features_for_monday(
            telemetry, gateway_master, meter_reads, field_visits,
            monday, baseline_days, recent_days,
        )
        if not features.empty:
            all_features.append(features)

    if not all_features:
        logger.error("No features built for any Monday")
        return pd.DataFrame()

    stacked = pd.concat(all_features, axis=0)
    logger.info("Total feature matrix: %d rows × %d columns", len(stacked), len(stacked.columns))
    return stacked


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding metadata columns)."""
    metadata_cols = {"_monday", "gateway_id", "label"}
    return [c for c in feature_df.columns if c not in metadata_cols]
