"""
End-to-end pipeline orchestrator — connects data loading, feature engineering,
model training/prediction, and output generation.
"""

from __future__ import annotations

import logging
import pathlib

import pandas as pd

from src.config import Config
from src.data.loader import load_all
from src.data.validator import validate_all
from src.features.builder import build_features_for_all_weeks, get_feature_columns
from src.model.predictor import load_model, predict
from src.model.ranker import rank_predictions
from src.model.trainer import create_labels, train_model, save_model
from src.monitoring.data_drift import DataDriftMonitor

logger = logging.getLogger(__name__)


def run_training_pipeline(config: Config) -> pathlib.Path:
    """Full training pipeline: load → validate → featurize → train → save.

    Returns:
        Path to the saved model version directory.
    """
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 70)

    # 1. Load data
    datasets = load_all(config.data_dir)

    # 2. Validate
    validation = validate_all(datasets)
    if not validation.passed:
        raise RuntimeError(f"Data validation failed:\n{validation.summary()}")

    # 3. Build features for training weeks
    # Use weeks before the scored window for training
    training_mondays = [
        "2025-10-06", "2025-10-13", "2025-10-20", "2025-10-27",
        "2025-11-03", "2025-11-10", "2025-11-17", "2025-11-24",
        "2025-12-01", "2025-12-08", "2025-12-15", "2025-12-22",
        "2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26",
    ]

    features = build_features_for_all_weeks(
        datasets["telemetry"],
        datasets["gateway_master"],
        datasets["meter_read_success"],
        datasets["field_visits"],
        training_mondays,
        baseline_days=config.baseline_days,
        recent_days=config.recent_days,
    )

    # 4. Create labels
    labels = create_labels(
        features, datasets["engineer_review"],
        datasets["field_visits"], datasets["meter_read_success"],
    )

    # 5. Train model
    feature_columns = get_feature_columns(features)
    model, metadata = train_model(features, labels, config, feature_columns)

    # 6. Save model
    config.model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = save_model(model, metadata, config.model_dir)

    # 7. Save drift reference
    drift_monitor = DataDriftMonitor(drift_threshold=config.drift_threshold)
    drift_monitor.fit_reference(datasets["telemetry"])
    drift_monitor.save_reference(version_dir / "drift_reference.json")

    logger.info("TRAINING PIPELINE COMPLETE → %s", version_dir)
    return version_dir


def run_prediction_pipeline(config: Config) -> pd.DataFrame:
    """Full prediction pipeline: load → featurize → predict → rank → save.

    Returns:
        Predictions DataFrame.
    """
    logger.info("=" * 70)
    logger.info("PREDICTION PIPELINE START")
    logger.info("=" * 70)

    # 1. Load data
    datasets = load_all(config.data_dir)

    # 2. Validate
    validation = validate_all(datasets)
    if not validation.passed:
        logger.warning("Data validation issues:\n%s", validation.summary())

    # 3. Load model
    model, metadata = load_model(config.model_dir, config.model_version)

    # 4. Optional: check for drift
    try:
        version = config.model_version
        if version == "current":
            version = (config.model_dir / "current_version.txt").read_text().strip()
        drift_ref_path = config.model_dir / version / "drift_reference.json"
        if drift_ref_path.exists():
            drift_monitor = DataDriftMonitor(drift_threshold=config.drift_threshold)
            drift_monitor.load_reference(drift_ref_path)
            drift_report = drift_monitor.check(datasets["telemetry"])
            if not drift_report.is_ok:
                logger.warning("DATA DRIFT DETECTED:\n%s", drift_report.summary())
    except Exception as e:
        logger.warning("Drift check skipped: %s", e)

    # 5. Build features for scored weeks
    scored_mondays = list(config.scored_weeks)

    features = build_features_for_all_weeks(
        datasets["telemetry"],
        datasets["gateway_master"],
        datasets["meter_read_success"],
        datasets["field_visits"],
        scored_mondays,
        baseline_days=config.baseline_days,
        recent_days=config.recent_days,
    )

    # 6. Generate predictions
    predictions = predict(model, features, metadata)

    # 7. Rank and format
    ranked = rank_predictions(
        predictions, features,
        budget=config.visit_budget,
        cost_fp=config.cost_fp,
        cost_fn=config.cost_fn,
    )

    # 8. Save predictions
    ranked.to_csv(config.predictions_out, index=False)
    logger.info("Predictions saved to %s — %d rows over %d weeks",
                config.predictions_out, len(ranked), ranked["week_start"].nunique())

    logger.info("PREDICTION PIPELINE COMPLETE")
    return ranked
