#!/usr/bin/env python3
"""Benchmark and compare multiple ML models for gateway health prediction.

Compares:
1. LightGBM (Gradient Boosted Trees with cost-sensitive weighting)
2. HistGradientBoosting (Histogram-based GBDT)
3. RandomForest (Bagged decision trees with median imputation)
4. LogisticRegression (L2-regularised linear model with imputation & scaling)

Usage:
    python scripts/compare_models.py --data ./data
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import load_all
from src.data.validator import validate_all
from src.features.builder import build_features_for_all_weeks, get_feature_columns
from src.model.trainer import (
    create_labels,
    _train_lightgbm,
    _train_hist_gradient_boosting,
    _train_random_forest,
    _train_logistic_regression,
)


def run_benchmark(data_dir: pathlib.Path, random_seed: int = 42) -> pd.DataFrame:
    """Run cross-validated benchmark comparing all 4 model architectures."""
    print("=" * 80)
    print("GATEWAY HEALTH PREDICTION — MULTI-MODEL BENCHMARK")
    print("=" * 80)

    config = Config(data_dir=data_dir, random_seed=random_seed)
    datasets = load_all(config.data_dir)
    val = validate_all(datasets)
    if not val.passed:
        raise RuntimeError(f"Data validation failed:\n{val.summary()}")

    training_mondays = [
        "2025-10-06", "2025-10-13", "2025-10-20", "2025-10-27",
        "2025-11-03", "2025-11-10", "2025-11-17", "2025-11-24",
        "2025-12-01", "2025-12-08", "2025-12-15", "2025-12-22",
        "2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26",
    ]

    print("\nBuilding feature matrix across 16 training Mondays...")
    t0 = time.time()
    features = build_features_for_all_weeks(
        datasets["telemetry"],
        datasets["gateway_master"],
        datasets["meter_read_success"],
        datasets["field_visits"],
        training_mondays,
        baseline_days=config.baseline_days,
        recent_days=config.recent_days,
    )
    labels = create_labels(
        features,
        datasets["engineer_review"],
        datasets["field_visits"],
        datasets["meter_read_success"],
    )
    feature_cols = get_feature_columns(features)
    print(f"Features built in {time.time() - t0:.1f}s: {len(features)} samples × {len(feature_cols)} features\n")

    models = [
        ("LightGBM (GBDT)", "lightgbm", _train_lightgbm),
        ("HistGradientBoosting", "hist_gradient_boosting", _train_hist_gradient_boosting),
        ("RandomForest", "random_forest", _train_random_forest),
        ("LogisticRegression", "logistic_regression", _train_logistic_regression),
    ]

    records = []
    for display_name, m_type, train_fn in models:
        print(f"--> Training & 5-Fold Cross-Validating {display_name}...")
        t_start = time.time()
        model, metadata = train_fn(features, labels, config, feature_cols)
        t_elapsed = time.time() - t_start

        records.append({
            "Model": display_name,
            "Type": m_type,
            "CV Cost (EUR)": metadata.get("cv_total_cost_eur", float("nan")),
            "ROC-AUC": metadata.get("cv_auc", float("nan")),
            "Log-Loss": metadata.get("cv_log_loss", float("nan")),
            "Fit Time (s)": round(t_elapsed, 2),
            "Cost Penalty vs LightGBM": "N/A",
        })

    df = pd.DataFrame(records)
    lgb_cost = df.loc[df["Type"] == "lightgbm", "CV Cost (EUR)"].values[0]
    df["Cost Penalty vs LightGBM"] = df["CV Cost (EUR)"].apply(
        lambda c: f"+EUR {c - lgb_cost:,.0f}" if c > lgb_cost else "Baseline (Best)"
    )

    print("\n" + "=" * 88)
    print(f"{'Model':<25} | {'5-Fold CV Cost':<16} | {'ROC-AUC':<8} | {'Log-Loss':<8} | {'Time (s)':<8} | {'vs Best':<14}")
    print("-" * 88)
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} | EUR {row['CV Cost (EUR)']:<12.1f} | {row['ROC-AUC']:<8.4f} | {row['Log-Loss']:<8.4f} | {row['Fit Time (s)']:<8.2f} | {row['Cost Penalty vs LightGBM']:<14}")
    print("=" * 88)

    # Save benchmark results
    out_dir = config.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_comparison.json"
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nBenchmark results saved to {out_file}\n")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multiple ML models")
    parser.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "./data"),
                        help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_benchmark(pathlib.Path(args.data), random_seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
