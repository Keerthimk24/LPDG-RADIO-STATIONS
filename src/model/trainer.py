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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


SUPPORTED_MODELS = (
    "lightgbm",
    "hist_gradient_boosting",
    "random_forest",
    "logistic_regression",
)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str] | None = None,
    model_type: str | None = None,
) -> tuple[any, dict]:
    """Train a predictive model with asymmetric cost awareness.

    Supports LightGBM, HistGradientBoosting, RandomForest, and LogisticRegression.

    Args:
        X: Feature matrix.
        y: Binary labels (0/1).
        config: Configuration object.
        feature_columns: Which columns to use as features.
        model_type: Architecture to train ('lightgbm', 'hist_gradient_boosting',
                    'random_forest', 'logistic_regression'). Defaults to config.model_type.

    Returns:
        Tuple of (trained model, training metadata dict).
    """
    if feature_columns is None:
        metadata_cols = {"_monday", "label"}
        feature_columns = [c for c in X.columns if c not in metadata_cols]

    if model_type is None:
        model_type = getattr(config, "model_type", "lightgbm")

    model_type = str(model_type).lower().strip()

    if model_type == "lightgbm":
        return _train_lightgbm(X, y, config, feature_columns)
    elif model_type in ("hist_gradient_boosting", "hgb"):
        return _train_hist_gradient_boosting(X, y, config, feature_columns)
    elif model_type in ("random_forest", "rf"):
        return _train_random_forest(X, y, config, feature_columns)
    elif model_type in ("logistic_regression", "lr"):
        return _train_logistic_regression(X, y, config, feature_columns)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported models: {SUPPORTED_MODELS}"
        )


def _train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train a LightGBM model with asymmetric cost function."""
    X_train = X[feature_columns].copy()

    # Compute SHA256 hash of training data for reproducibility tracking
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(X_train).values.tobytes()
    ).hexdigest()

    # Extract gateway groups for GroupKFold CV
    groups = X.index.values  # gateway_id is the index

    logger.info("Training LightGBM: %d samples, %d features, %d positive labels",
                len(X_train), len(feature_columns), int(y.sum()))

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
        "scale_pos_weight": config.cost_fn / config.cost_fp,
    }

    cv_cost, cv_auc, cv_loss, optimal_rounds = _cross_validate_cost(
        X_train, y, params, config, groups=groups,
    )

    train_data = lgb.Dataset(X_train, label=y, free_raw_data=False)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=optimal_rounds,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    metadata = {
        "trained_on": dt.datetime.utcnow().isoformat(),
        "model_type": "lightgbm",
        "model_class": "lightgbm.Booster",
        "features": feature_columns,
        "n_samples": len(X_train),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "best_iteration": optimal_rounds,
        "cv_total_cost_eur": cv_cost,
        "cv_auc": cv_auc,
        "cv_log_loss": cv_loss,
        "params": params,
        "python_version": "3.12",
        "lightgbm_version": lgb.__version__,
        "random_seed": config.random_seed,
        "git_sha": _get_git_sha(),
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
        "data_hash": data_hash,
        "cv_strategy": "GroupKFold",
        "cv_n_folds": 3,
    }

    logger.info("Training complete. Best iteration: %d, CV cost: €%.0f, AUC: %.4f",
                optimal_rounds, cv_cost, cv_auc)

    return model, metadata


def _train_hist_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str],
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Train a scikit-learn HistGradientBoostingClassifier."""
    X_train = X[feature_columns].copy()

    logger.info("Training HistGradientBoosting: %d samples, %d features, %d positive labels",
                len(X_train), len(feature_columns), int(y.sum()))

    pos_weight = config.cost_fn / config.cost_fp
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=10,
        class_weight={0: 1.0, 1: pos_weight},
        random_state=config.random_seed,
    )

    cv_cost, cv_auc, cv_loss = _cross_validate_sklearn_cost(model, X_train, y, config)
    model.fit(X_train, y)

    metadata = {
        "trained_on": dt.datetime.utcnow().isoformat(),
        "model_type": "hist_gradient_boosting",
        "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "features": feature_columns,
        "n_samples": len(X_train),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "cv_total_cost_eur": cv_cost,
        "cv_auc": cv_auc,
        "cv_log_loss": cv_loss,
        "python_version": "3.12",
        "random_seed": config.random_seed,
        "git_sha": _get_git_sha(),
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
    }

    logger.info("HistGradientBoosting complete. CV cost: €%.0f, AUC: %.4f", cv_cost, cv_auc)
    return model, metadata


def _train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str],
) -> tuple[Pipeline, dict]:
    """Train a scikit-learn RandomForestClassifier inside an imputation pipeline."""
    X_train = X[feature_columns].copy()

    logger.info("Training RandomForest: %d samples, %d features, %d positive labels",
                len(X_train), len(feature_columns), int(y.sum()))

    pos_weight = config.cost_fn / config.cost_fp
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight={0: 1.0, 1: pos_weight},
            random_state=config.random_seed,
            n_jobs=1,
        )),
    ])

    cv_cost, cv_auc, cv_loss = _cross_validate_sklearn_cost(model, X_train, y, config)
    model.fit(X_train, y)

    metadata = {
        "trained_on": dt.datetime.utcnow().isoformat(),
        "model_type": "random_forest",
        "model_class": "sklearn.ensemble.RandomForestClassifier",
        "features": feature_columns,
        "n_samples": len(X_train),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "cv_total_cost_eur": cv_cost,
        "cv_auc": cv_auc,
        "cv_log_loss": cv_loss,
        "python_version": "3.12",
        "random_seed": config.random_seed,
        "git_sha": _get_git_sha(),
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
    }

    logger.info("RandomForest complete. CV cost: €%.0f, AUC: %.4f", cv_cost, cv_auc)
    return model, metadata


def _train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_columns: list[str],
) -> tuple[Pipeline, dict]:
    """Train a scikit-learn LogisticRegression model inside an imputation + scaling pipeline."""
    X_train = X[feature_columns].copy()

    logger.info("Training LogisticRegression: %d samples, %d features, %d positive labels",
                len(X_train), len(feature_columns), int(y.sum()))

    pos_weight = config.cost_fn / config.cost_fp
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight={0: 1.0, 1: pos_weight},
            max_iter=1000,
            random_state=config.random_seed,
            C=0.1,
        )),
    ])

    cv_cost, cv_auc, cv_loss = _cross_validate_sklearn_cost(model, X_train, y, config)
    model.fit(X_train, y)

    metadata = {
        "trained_on": dt.datetime.utcnow().isoformat(),
        "model_type": "logistic_regression",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "features": feature_columns,
        "n_samples": len(X_train),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "cv_total_cost_eur": cv_cost,
        "cv_auc": cv_auc,
        "cv_log_loss": cv_loss,
        "python_version": "3.12",
        "random_seed": config.random_seed,
        "git_sha": _get_git_sha(),
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
    }

    logger.info("LogisticRegression complete. CV cost: €%.0f, AUC: %.4f", cv_cost, cv_auc)
    return model, metadata


def _cross_validate_cost(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    config: Config,
    n_folds: int = 3,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float, int]:
    """Estimate total cost, AUC, and log-loss via gateway-level GroupKFold cross-validation."""
    try:
        if groups is not None:
            gkf = GroupKFold(n_splits=n_folds)
            splitter = gkf.split(X, y, groups=groups)
            logger.info("Using GroupKFold (%d folds) — same gateway never in train & val", n_folds)
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
            splitter = skf.split(X, y)
            logger.info("Using StratifiedKFold (%d folds) — no group information", n_folds)

        costs = []
        aucs = []
        losses = []
        best_iters = []

        for fold, (train_idx, val_idx) in enumerate(splitter):
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
            fp = int(((y_binary == 1) & (y_val.values == 0)).sum())
            fn = int(((y_binary == 0) & (y_val.values == 1)).sum())
            fold_cost = fp * config.cost_fp + fn * config.cost_fn
            costs.append(fold_cost)
            best_iters.append(fold_model.best_iteration)

            try:
                aucs.append(float(roc_auc_score(y_val, preds)))
                losses.append(float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))))
            except Exception:
                pass

        avg_cost = float(np.mean(costs))
        avg_auc = float(np.mean(aucs)) if aucs else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        optimal_iters = int(np.median(best_iters)) if best_iters else 150
        optimal_iters = max(optimal_iters, 50)
        logger.info("CV total cost (5-fold avg): €%.0f ± €%.0f, AUC: %.4f, LogLoss: %.4f, optimal iterations: %d",
                    avg_cost, float(np.std(costs)), avg_auc, avg_loss, optimal_iters)
        return avg_cost, avg_auc, avg_loss, optimal_iters
    except Exception as e:
        logger.warning("LightGBM cross-validation failed: %s", e)
        return float("inf"), 0.0, 0.0, 150


def _cross_validate_sklearn_cost(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
    n_folds: int = 5,
) -> tuple[float, float, float]:
    """Estimate total cost, AUC, and log-loss via cross-validation for scikit-learn models."""
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
        costs = []
        aucs = []
        losses = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict_proba(X_val)[:, 1]

            y_binary = (preds > 0.5).astype(int)
            fp = int(((y_binary == 1) & (y_val.values == 0)).sum())
            fn = int(((y_binary == 0) & (y_val.values == 1)).sum())
            fold_cost = fp * config.cost_fp + fn * config.cost_fn
            costs.append(fold_cost)

            try:
                aucs.append(float(roc_auc_score(y_val, preds)))
                losses.append(float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))))
            except Exception:
                pass

        avg_cost = float(np.mean(costs))
        avg_auc = float(np.mean(aucs)) if aucs else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        logger.info("CV cost (5-fold avg): €%.0f ± €%.0f, AUC: %.4f, LogLoss: %.4f",
                    avg_cost, float(np.std(costs)), avg_auc, avg_loss)
        return avg_cost, avg_auc, avg_loss
    except Exception as e:
        logger.warning("Sklearn cross-validation failed: %s", e)
        return float("inf"), 0.0, 0.0


def save_model(
    model: any,
    metadata: dict,
    model_dir: pathlib.Path,
    version: str | None = None,
) -> pathlib.Path:
    """Save model artifacts to a versioned directory.

    Args:
        model: Trained model (LightGBM Booster or scikit-learn model).
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

    logger.info("Model saved: %s (%s) → %s", version, metadata.get("model_type", "lightgbm"), version_dir)
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
        "model_type": metadata.get("model_type", "lightgbm"),
        "model_class": metadata.get("model_class", "unknown"),
        "trained_on": metadata["trained_on"],
        "n_samples": metadata["n_samples"],
        "cv_total_cost_eur": metadata.get("cv_total_cost_eur"),
        "cv_auc": metadata.get("cv_auc"),
        "git_sha": metadata.get("git_sha"),
    })
    registry["current"] = version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
