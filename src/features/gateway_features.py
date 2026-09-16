"""
Gateway static features — metadata from the asset register.

These are time-invariant features (hw model, firmware, region, etc.) that provide
context about WHY a gateway might be at risk.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_gateway_features(
    gateway_master: pd.DataFrame,
    monday: str | dt.date,
) -> pd.DataFrame:
    """Build static gateway features for a prediction Monday.

    Args:
        gateway_master: Gateway asset register DataFrame.
        monday: The prediction Monday (used for age calculations).

    Returns:
        DataFrame indexed by gateway_id with static features.
    """
    if isinstance(monday, str):
        monday = dt.date.fromisoformat(monday)

    monday_ts = pd.Timestamp(monday)

    # Filter to active gateways only (not decommissioned before this Monday)
    active = gateway_master[
        gateway_master["decommissioned_on"].isna() |
        (gateway_master["decommissioned_on"] >= monday_ts)
    ].copy()

    features = pd.DataFrame(index=active["gateway_id"])
    features.index.name = "gateway_id"

    # Number of meters (higher = higher impact if gateway fails)
    features["n_meters_installed"] = active.set_index("gateway_id")["n_meters_installed"]

    # Gateway age in days
    installed = active.set_index("gateway_id")["installed_on"]
    features["gateway_age_days"] = (monday_ts - installed).dt.days.fillna(0).clip(lower=0)

    # Days since firmware update (NaN if never updated → very old firmware)
    fw_updated = active.set_index("gateway_id")["fw_updated_on"]
    features["days_since_fw_update"] = (monday_ts - fw_updated).dt.days.fillna(9999).clip(lower=0)

    # One-hot encode categorical features
    categoricals = {
        "hw_model": active.set_index("gateway_id")["hw_model"],
        "antenna_type": active.set_index("gateway_id")["antenna_type"],
        "tenant": active.set_index("gateway_id")["tenant"],
        "region": active.set_index("gateway_id")["region"],
    }

    for cat_name, cat_series in categoricals.items():
        dummies = pd.get_dummies(cat_series, prefix=cat_name, dtype=int)
        features = features.join(dummies)

    # Firmware version as ordinal (older = potentially riskier)
    fw_map = {"2.14.3": 0, "2.15.1": 1, "3.2.0": 2, "3.3.1": 3}
    fw_series = active.set_index("gateway_id")["fw_version"]
    features["fw_version_ordinal"] = fw_series.map(fw_map).fillna(-1).astype(int)

    features = features.fillna(0)

    logger.info("Built gateway features: %d gateways, %d features",
                len(features), len(features.columns))
    return features
