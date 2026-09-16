#!/usr/bin/env python3
"""Train the gateway health prediction model.

Usage:
    python scripts/train.py --data ./data --model-dir ./models
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.pipeline import run_training_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the gateway health prediction model")
    parser.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "./data"),
                        help="Path to data directory")
    parser.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", "./models"),
                        help="Path to save model artifacts")
    parser.add_argument("--model-type", type=str, default=os.getenv("MODEL_TYPE", "lightgbm"),
                        choices=["lightgbm", "hist_gradient_boosting", "random_forest", "logistic_regression"],
                        help="Architecture to train (default: lightgbm)")
    parser.add_argument("--compare", action="store_true",
                        help="Train and benchmark all supported models side-by-side")
    args = parser.parse_args(argv)

    # Override env vars with CLI args
    os.environ["DATA_DIR"] = args.data
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["MODEL_TYPE"] = args.model_type

    config = Config()
    try:
        config.validate()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.compare:
        print("\n" + "=" * 75)
        print("BENCHMARKING ALL MODELS (5-Fold Cross-Validation)")
        print("=" * 75)
        model_types = ["lightgbm", "hist_gradient_boosting", "random_forest", "logistic_regression"]
        results = []
        for m_type in model_types:
            print(f"\n--- Training {m_type} ---")
            os.environ["MODEL_TYPE"] = m_type
            cfg = Config()
            v_dir = run_training_pipeline(cfg)
            meta_path = v_dir / "metadata.json"
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            results.append({
                "model_type": m_type,
                "version": v_dir.name,
                "cost_eur": meta.get("cv_total_cost_eur", float("nan")),
                "auc": meta.get("cv_auc", float("nan")),
                "log_loss": meta.get("cv_log_loss", float("nan")),
            })

        print("\n" + "=" * 75)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 75)
        print(f"{'Model':<24} | {'Version':<8} | {'5-Fold CV Cost':<16} | {'ROC-AUC':<8} | {'LogLoss':<8}")
        print("-" * 75)
        for r in sorted(results, key=lambda x: x["cost_eur"]):
            print(f"{r['model_type']:<24} | {r['version']:<8} | EUR {r['cost_eur']:<12.1f} | {r['auc']:<8.4f} | {r['log_loss']:<8.4f}")
        print("=" * 75 + "\n")
        return 0

    try:
        version_dir = run_training_pipeline(config)
        print(f"\nTraining complete ({config.model_type}) -> {version_dir}")
        return 0
    except Exception as e:
        print(f"ERROR: Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
