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


def compute_global_importance(model: any, feature_names: list[str]) -> pd.DataFrame:
    """Compute global feature importance from the trained model.

    Supports LightGBM, RandomForest, HistGradientBoosting, and LogisticRegression.

    Args:
        model: Trained model (LightGBM Booster or scikit-learn estimator/Pipeline).
        feature_names: List of feature column names.

    Returns:
        DataFrame with feature name and importance, sorted descending.
    """
    if hasattr(model, "feature_importance"):
        # LightGBM Booster
        importance = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps"):
        # Pipeline
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "feature_importances_"):
            importance = np.asarray(last_step.feature_importances_, dtype=float)
        elif hasattr(last_step, "coef_"):
            importance = np.abs(np.asarray(last_step.coef_, dtype=float).ravel())
        else:
            importance = np.ones(len(feature_names), dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        importance = np.ones(len(feature_names), dtype=float)

    total = importance.sum()
    pct = (importance / total * 100).round(2) if total > 0 else np.zeros_like(importance)

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "importance_pct": pct,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Top 10 features by importance:\n%s", result.head(10).to_string(index=False))
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
