#!/usr/bin/env python3
"""Generate predictions using a trained model.

Usage:
    python scripts/predict.py --data ./data --out predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.pipeline import run_prediction_pipeline, run_training_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate gateway visit predictions")
    parser.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "./data"),
                        help="Path to data directory")
    parser.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", "./models"),
                        help="Path to model artifacts")
    parser.add_argument("--out", type=str, default=os.getenv("PREDICTIONS_OUT", "./predictions.csv"),
                        help="Output predictions file path")
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "current"),
                        help="Model version to use (default: current)")
    args = parser.parse_args(argv)

    # Override env vars with CLI args
    os.environ["DATA_DIR"] = args.data
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["PREDICTIONS_OUT"] = args.out
    os.environ["MODEL_VERSION"] = args.model_version

    config = Config()
    try:
        config.validate()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Auto-train if no model exists yet (enables true single-command execution)
    import pathlib
    model_marker = pathlib.Path(args.model_dir) / "current_version.txt"
    if not model_marker.exists():
        print("No trained model found. Training automatically before predicting...")
        try:
            version_dir = run_training_pipeline(config)
            print(f"Training complete -> {version_dir}")
        except Exception as e:
            print(f"ERROR: Auto-training failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    try:
        predictions = run_prediction_pipeline(config)
        print(f"\nPredictions written to {args.out}")
        print(f"  {len(predictions)} rows over {predictions['week_start'].nunique()} weeks")
        return 0
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
