"""
Visit history features — patterns from past field visits.

Key insight: 61% of visits found nothing wrong. Gateways that have been visited
before and had actual faults are more likely to need visits again.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_visit_features(
    field_visits: pd.DataFrame,
    monday: str | dt.date,
) -> pd.DataFrame:
    """Build visit history features for a prediction Monday.

    Args:
        field_visits: Historical field visit work orders.
        monday: The prediction Monday.

    Returns:
        DataFrame indexed by gateway_id with visit history features.
    """
    if isinstance(monday, str):
        monday = dt.date.fromisoformat(monday)

    monday_ts = pd.Timestamp(monday)

    # Only use visits that happened strictly before this Monday
    past_visits = field_visits[field_visits["visited_on"] < monday_ts].copy()

    if past_visits.empty:
        logger.warning("No visit history before %s", monday)
        return pd.DataFrame()

    grouped = past_visits.groupby("gateway_id")

    features = pd.DataFrame()

    # Total visit count
    features["total_visits"] = grouped.size()

    # Days since last visit
    last_visit = grouped["visited_on"].max()
    features["days_since_last_visit"] = (monday_ts - last_visit).dt.days

    # Last visit outcome
    def get_last_outcome(group):
        latest = group.sort_values("visited_on").iloc[-1]
        return latest["outcome"]

    last_outcomes = grouped.apply(get_last_outcome)
    features["last_visit_found_fault"] = (last_outcomes == "Fehler behoben").astype(int)
    features["last_visit_no_access"] = (last_outcomes == "Kein Zugang").astype(int)

    # Fault rate: how often visits actually found something wrong
    def fault_rate(group):
        return (group["outcome"] == "Fehler behoben").mean()

    features["visit_fault_rate"] = grouped.apply(fault_rate)

    # Has had parts replaced
    features["had_parts_replaced"] = grouped["parts_replaced"].apply(
        lambda s: s.notna().any()
    ).astype(int)

    # Average technician hours (proxy for severity)
    features["avg_technician_hours"] = grouped["technician_hours"].mean()

    # Repeat offender: more than 2 visits
    features["is_repeat_offender"] = (features["total_visits"] > 2).astype(int)

    # Visit reasons (counts of each reason type)
    reason_counts = past_visits.groupby(["gateway_id", "reason_reported"]).size().unstack(fill_value=0)
    reason_counts.columns = [f"visit_reason_{c.replace(' ', '_')}" for c in reason_counts.columns]
    features = features.join(reason_counts, how="left")

    features.index.name = "gateway_id"
    features = features.fillna(0)

    logger.info("Built visit features: %d gateways, %d features",
                len(features), len(features.columns))
    return features
