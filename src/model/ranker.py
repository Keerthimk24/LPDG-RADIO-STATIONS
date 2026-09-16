"""
Ranker — converts model probabilities to a ranked list of top-15 gateways
per week with human-readable reasons.

Output matches the predictions.csv schema:
  week_start, rank, gateway_id, score, reason
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.model.cost_function import rank_by_cost

logger = logging.getLogger(__name__)


def format_gateway_id(gw_id: str) -> str:
    """Format gateway ID as 12-char uppercase hex (no colons)."""
    return re.sub(r"[^0-9A-Fa-f]", "", str(gw_id)).upper()


def rank_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    budget: int = 15,
    cost_fp: float = 380.0,
    cost_fn: float = 600.0,
) -> pd.DataFrame:
    """Rank predictions and generate the final predictions.csv content.

    Args:
        predictions: DataFrame with gateway_id, probability, _monday.
        features: Feature matrix (for generating reasons).
        budget: Maximum visits per week.
        cost_fp: Cost of unnecessary visit.
        cost_fn: Cost of missed broken gateway.

    Returns:
        DataFrame in predictions.csv format.
    """
    all_rows = []

    for monday in sorted(predictions["_monday"].unique()):
        week_preds = predictions[predictions["_monday"] == monday]
        week_feats = features[features["_monday"] == monday] if "_monday" in features.columns else features

        # Rank by expected cost savings
        ranked = rank_by_cost(
            week_preds["gateway_id"].values,
            week_preds["probability"].values,
            budget=budget,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        for entry in ranked:
            gw_id = entry["gateway_id"]
            reason = _generate_reason(gw_id, entry, week_feats)

            all_rows.append({
                "week_start": monday,
                "rank": entry["rank"],
                "gateway_id": format_gateway_id(gw_id),
                "score": round(entry["score"], 4),
                "reason": reason[:300],  # Hard limit: 300 chars
            })

    result = pd.DataFrame(all_rows)
    logger.info("Ranked predictions: %d rows across %d weeks",
                len(result), result["week_start"].nunique())
    return result


def _generate_reason(
    gateway_id: str,
    ranking_entry: dict,
    features: pd.DataFrame,
) -> str:
    """Generate a human-readable reason for why this gateway was ranked here.

    Written for the operations manager, not for data scientists.
    Max 300 characters.
    """
    prob = ranking_entry["probability"]
    rank = ranking_entry["rank"]

    # Try to get gateway-specific feature values
    parts = []

    if gateway_id in features.index:
        row = features.loc[gateway_id]

        # Most impactful telemetry signals
        offline_mean = row.get("7d_offline_duration_sec_mean", 0)
        offline_trend = row.get("trend_offline_duration_sec_ratio", 1)
        disc_mean = row.get("7d_disconnection_cnt_mean", 0)
        disc_trend = row.get("trend_disconnection_cnt_ratio", 1)
        reboot_mean = row.get("7d_reboot_cnt_mean", 0)
        reboot_trend = row.get("trend_reboot_cnt_ratio", 1)
        meter_rate = row.get("meter_read_rate_latest", 1)
        meters_at_risk = row.get("meters_at_risk", 0)
        n_meters = row.get("n_meters_installed", 0)

        # Build reason from most alarming signals
        if offline_trend > 1.5:
            parts.append(f"offline time {offline_trend:.1f}x above 28d baseline")
        elif offline_mean > 100:
            parts.append(f"avg {offline_mean:.0f}s offline/hour")

        if disc_trend > 1.5:
            parts.append(f"disconnections {disc_trend:.1f}x above baseline")
        elif disc_mean > 0.5:
            parts.append(f"{disc_mean:.1f} disconnections/hour avg")

        if reboot_trend > 2:
            parts.append(f"reboots {reboot_trend:.1f}x above baseline")
        elif reboot_mean > 0.1:
            parts.append(f"{reboot_mean:.2f} reboots/hour avg")

        if meter_rate < 0.8:
            parts.append(f"meter read rate {meter_rate:.0%}")

        if meters_at_risk > 20:
            parts.append(f"{meters_at_risk:.0f} meters at risk")

        if n_meters > 200 and not parts:
            parts.append(f"high-impact gateway ({n_meters:.0f} meters)")

    if not parts:
        parts.append(f"risk score {prob:.2f}")

    reason = f"Rank #{rank}: " + "; ".join(parts)
    return reason
