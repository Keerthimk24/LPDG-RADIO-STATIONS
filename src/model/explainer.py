"""
Model explainer — SHAP-based feature importance and per-prediction explanations.

Provides:
1. Global feature importance (what the model pays attention to overall)
2. Per-gateway explanations (why THIS gateway was ranked here)
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_global_importance(model: lgb.Booster, feature_names: list[str]) -> pd.DataFrame:
    """Compute global feature importance from the trained model.

    Args:
        model: Trained LightGBM model.
        feature_names: List of feature column names.

    Returns:
        DataFrame with feature name and importance, sorted descending.
    """
    importance = model.feature_importance(importance_type="gain")

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    result["importance_pct"] = (result["importance"] / result["importance"].sum() * 100).round(2)

    logger.info("Top 10 features by gain:\n%s", result.head(10).to_string(index=False))
    return result


def compute_shap_values(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Compute SHAP values for a set of predictions.

    Args:
        model: Trained LightGBM model.
        X: Feature matrix (samples × features).
        feature_names: Feature column names.

    Returns:
        SHAP values array (samples × features).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_names])

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (positive = needs visit)

    logger.info("SHAP values computed: %s", shap_values.shape)
    return shap_values


def explain_prediction(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    top_n: int = 3,
) -> list[dict]:
    """Explain a single prediction using its SHAP values.

    Args:
        shap_values: SHAP values for one sample.
        feature_names: Feature column names.
        feature_values: Feature values for this sample.
        top_n: Number of top contributing features.

    Returns:
        List of dicts with feature, shap_value, feature_value, direction.
    """
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    explanations = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        shap_val = float(shap_values[idx])
        feat_val = float(feature_values.get(feat_name, 0))
        direction = "increases risk" if shap_val > 0 else "decreases risk"

        explanations.append({
            "feature": feat_name,
            "shap_value": shap_val,
            "feature_value": feat_val,
            "direction": direction,
        })

    return explanations
