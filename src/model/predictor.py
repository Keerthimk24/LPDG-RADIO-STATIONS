"""
Model predictor — loads a trained model and generates predictions.

Genuinely separate from training (MLOps requirement).
Same inputs and same version give the same answer (reproducibility).
"""

from __future__ import annotations

import json
import logging
import pathlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_dir: pathlib.Path, version: str = "current") -> tuple[lgb.Booster, dict]:
    """Load a trained model and its metadata.

    Args:
        model_dir: Base models directory.
        version: Version string or "current" for the latest.

    Returns:
        Tuple of (model, metadata dict).
    """
    if version == "current":
        current_marker = model_dir / "current_version.txt"
        if not current_marker.exists():
            raise FileNotFoundError(
                f"No current model version found. Run training first. "
                f"Expected: {current_marker}"
            )
        version = current_marker.read_text().strip()

    version_dir = model_dir / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Model version '{version}' not found: {version_dir}")

    model_path = version_dir / "model.joblib"
    model = joblib.load(model_path)

    meta_path = version_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info("Loaded model %s (trained on %s, %d features)",
                version, metadata.get("trained_on", "unknown"), len(metadata.get("features", [])))

    return model, metadata


def predict(
    model: lgb.Booster,
    features: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Generate predictions using a trained model.

    Args:
        model: Trained LightGBM model.
        features: Feature matrix (must contain the columns the model expects).
        metadata: Model metadata (for feature column validation).

    Returns:
        DataFrame with gateway_id, probability, and _monday columns.
    """
    expected_features = metadata.get("features", [])

    # Validate feature columns are present
    missing = [f for f in expected_features if f not in features.columns]
    if missing:
        logger.warning("Missing %d features for model prediction: %s", len(missing), missing[:5])
        # Add missing columns with zeros
        for col in missing:
            features[col] = 0

    X = features[expected_features].copy()

    # LightGBM binary objective returns calibrated probabilities directly
    probabilities = model.predict(X)

    result = pd.DataFrame({
        "gateway_id": features.index,
        "probability": probabilities,
    })

    if "_monday" in features.columns:
        result["_monday"] = features["_monday"].values

    logger.info("Predictions generated: %d gateways, prob range [%.3f, %.3f]",
                len(result), probabilities.min(), probabilities.max())

    return result
