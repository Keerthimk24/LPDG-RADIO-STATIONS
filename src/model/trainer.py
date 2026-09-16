"""
Model trainer — trains LightGBM with custom asymmetric cost function.

Training and predicting are genuinely separate steps (MLOps requirement).
The trained model is saved as a versioned file with metadata.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pathlib
import subprocess

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.model.cost_function import asymmetric_cost_objective, cost_metric

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def create_labels(
    features: pd.DataFrame,
    engineer_review: pd.DataFrame,
    field_visits: pd.DataFrame,
    meter_reads: pd.DataFrame,
) -> pd.Series:
    """Construct training labels using multi-signal approach.

    Primary signal: engineer review (Normal=0, Schlecht=1)
    Secondary signals: field visit outcomes, meter read collapse

    The feature matrix is stacked (same gateway_id appears for multiple Mondays),
    so we use vectorized operations on the gateway_id column, not the index.

    Args:
        features: Feature matrix with gateway_id as index (may have duplicates).
        engineer_review: Engineer review DataFrame with 'gateway_id' and 'label'.
        field_visits: Field visits with outcomes.
        meter_reads: Meter read success data.

    Returns:
        Series aligned with features index, binary labels (0=Normal, 1=Needs visit).
    """
    # Work with gateway_id from the index
    gw_ids = pd.Series(features.index, index=features.index, name="gateway_id")

    # Start with NaN
    labels = pd.Series(np.nan, index=features.index, name="label")

    # Signal 1: Engineer review (strongest — direct expert judgment)
    review_map = engineer_review.set_index("gateway_id")["label"].to_dict()
    review_signal = gw_ids.map(review_map)
    labels = labels.fillna(review_signal)

    # Signal 2: Field visits that found and fixed a fault → genuinely needed a visit
    fixed_visits = field_visits[field_visits["outcome"] == "Fehler behoben"]
    fixed_gws = set(fixed_visits["gateway_id"].unique())
    fixed_signal = gw_ids.apply(lambda g: 1.0 if g in fixed_gws else np.nan)
    labels = labels.fillna(fixed_signal)

    # Signal 3: Very poor meter read success (<50%) — proxy for broken gateway
    poor_reader_gws = set()
    if "_monday" in features.columns:
        for monday in features["_monday"].unique():
            monday_ts = pd.Timestamp(monday)
            recent_meter = meter_reads[
                (meter_reads["week_start"] >= monday_ts - pd.Timedelta(weeks=2)) &
                (meter_reads["week_start"] < monday_ts)
            ]
            poor = recent_meter[recent_meter["read_success_rate"] < 0.5]["gateway_id"].unique()
            poor_reader_gws.update(poor)

    if poor_reader_gws:
        poor_signal = gw_ids.apply(lambda g: 1.0 if g in poor_reader_gws else np.nan)
        labels = labels.fillna(poor_signal)

    # Signal 4: Gateways visited but nothing found → likely Normal
    no_fault_visits = field_visits[field_visits["outcome"] == "Kein Fehler gefunden"]
    no_fault_gws = set(no_fault_visits["gateway_id"].unique())
    no_fault_signal = gw_ids.apply(lambda g: 0.0 if g in no_fault_gws else np.nan)
    labels = labels.fillna(no_fault_signal)

    # Remaining unlabeled gateways: assume Normal (majority class)
    labels = labels.fillna(0.0)

    positive_rate = labels.mean()
    logger.info("Labels created: %d total, %d positive (%.1f%%)",
                len(labels), int(labels.sum()), positive_rate * 100)

    return labels


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str] | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train a LightGBM model with asymmetric cost function.

    Args:
        X: Feature matrix.
        y: Binary labels (0/1).
        config: Configuration object.
        feature_columns: Which columns to use as features.

    Returns:
        Tuple of (trained model, training metadata dict).
    """
    if feature_columns is None:
        metadata_cols = {"_monday", "label"}
        feature_columns = [c for c in X.columns if c not in metadata_cols]

    X_train = X[feature_columns].copy()

    logger.info("Training LightGBM: %d samples, %d features, %d positive labels",
                len(X_train), len(feature_columns), int(y.sum()))

    # LightGBM parameters — scale_pos_weight encodes cost asymmetry (€600 FN / €380 FP)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "seed": config.random_seed,
        "verbose": -1,
        "force_col_wise": True,
        # Cost-sensitive: scale_pos_weight reflects the cost asymmetry
        "scale_pos_weight": config.cost_fn / config.cost_fp,
    }

    # Cross-validation to estimate expected cost and find optimal iteration count
    cv_cost, optimal_rounds = _cross_validate_cost(X_train, y, params, config)

    train_data = lgb.Dataset(X_train, label=y, free_raw_data=False)

    # Train final model on all data with optimal number of iterations from CV
    model = lgb.train(
        params,
        train_data,
        num_boost_round=optimal_rounds,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    # Build metadata
    metadata = {
        "trained_on": dt.datetime.utcnow().isoformat(),
        "features": feature_columns,
        "n_samples": len(X_train),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "best_iteration": optimal_rounds,
        "cv_total_cost_eur": cv_cost,
        "params": params,
        "python_version": "3.12",
        "lightgbm_version": lgb.__version__,
        "random_seed": config.random_seed,
        "git_sha": _get_git_sha(),
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
    }

    logger.info("Training complete. Best iteration: %d, CV cost: €%.0f",
                optimal_rounds, cv_cost)

    return model, metadata


def _cross_validate_cost(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    config: Config,
    n_folds: int = 5,
) -> tuple[float, int]:
    """Estimate total cost via cross-validation and find optimal boost rounds."""
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
        costs = []
        best_iters = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            fold_model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )

            preds = fold_model.predict(X_val)
            y_binary = (preds > 0.5).astype(int)
            fp = ((y_binary == 1) & (y_val.values == 0)).sum()
            fn = ((y_binary == 0) & (y_val.values == 1)).sum()
            fold_cost = fp * config.cost_fp + fn * config.cost_fn
            costs.append(fold_cost)
            best_iters.append(fold_model.best_iteration)
            logger.debug("Fold %d cost: €%.0f (best iteration: %d)", fold, fold_cost, fold_model.best_iteration)

        avg_cost = np.mean(costs)
        optimal_iters = int(np.median(best_iters)) if best_iters else 150
        optimal_iters = max(optimal_iters, 50)
        logger.info("CV total cost (5-fold avg): €%.0f ± €%.0f, optimal iterations: %d",
                    avg_cost, np.std(costs), optimal_iters)
        return float(avg_cost), optimal_iters
    except Exception as e:
        logger.warning("Cross-validation failed: %s", e)
        return float("inf"), 150


def save_model(
    model: lgb.Booster,
    metadata: dict,
    model_dir: pathlib.Path,
    version: str | None = None,
) -> pathlib.Path:
    """Save model artifacts to a versioned directory.

    Args:
        model: Trained LightGBM model.
        metadata: Training metadata dict.
        model_dir: Base models directory.
        version: Version string (auto-generated if None).

    Returns:
        Path to the versioned model directory.
    """
    if version is None:
        # Auto-version: v{N+1}
        existing = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        version_nums = [int(v[1:]) for v in existing if v[1:].isdigit()]
        next_version = max(version_nums, default=0) + 1
        version = f"v{next_version}"

    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata["version"] = version
    meta_path = version_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save feature schema
    schema = {
        "columns": metadata["features"],
        "n_features": len(metadata["features"]),
    }
    schema_path = version_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Update current symlink (or file on Windows)
    current_marker = model_dir / "current_version.txt"
    with open(current_marker, "w") as f:
        f.write(version)

    # Update registry
    _update_registry(model_dir, version, metadata)

    logger.info("Model saved: %s → %s", version, version_dir)
    return version_dir


def _update_registry(model_dir: pathlib.Path, version: str, metadata: dict):
    """Update the model registry index."""
    registry_path = model_dir / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": [], "current": None}

    registry["models"].append({
        "version": version,
        "trained_on": metadata["trained_on"],
        "n_samples": metadata["n_samples"],
        "cv_total_cost_eur": metadata.get("cv_total_cost_eur"),
        "git_sha": metadata.get("git_sha"),
    })
    registry["current"] = version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
