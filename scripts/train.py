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
    args = parser.parse_args(argv)

    # Override env vars with CLI args
    os.environ["DATA_DIR"] = args.data
    os.environ["MODEL_DIR"] = args.model_dir

    config = Config()
    try:
        config.validate()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        version_dir = run_training_pipeline(config)
        print(f"\nTraining complete -> {version_dir}")
        return 0
    except Exception as e:
        print(f"ERROR: Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
