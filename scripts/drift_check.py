#!/usr/bin/env python3
"""Run data drift detection against the current model's reference distribution.

Usage:
    python scripts/drift_check.py --data ./data --model-dir ./models
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.loader import load_telemetry
from src.monitoring.data_drift import DataDriftMonitor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data drift detection")
    parser.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "./data"),
                        help="Path to data directory")
    parser.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", "./models"),
                        help="Path to model artifacts")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="KS-test p-value threshold (default: 0.05)")
    args = parser.parse_args(argv)

    model_dir = pathlib.Path(args.model_dir)
    data_dir = pathlib.Path(args.data)

    # Find current model version
    current_marker = model_dir / "current_version.txt"
    if not current_marker.exists():
        print("ERROR: No trained model found. Run training first.", file=sys.stderr)
        return 1

    version = current_marker.read_text().strip()
    drift_ref_path = model_dir / version / "drift_reference.json"

    if not drift_ref_path.exists():
        print(f"ERROR: No drift reference found at {drift_ref_path}", file=sys.stderr)
        return 1

    # Load current telemetry
    print(f"Loading telemetry from {data_dir}...")
    telemetry = load_telemetry(data_dir)

    # Run drift checks
    monitor = DataDriftMonitor(drift_threshold=args.threshold)
    monitor.load_reference(drift_ref_path)

    print(f"\nRunning drift checks against {version} reference...")
    report = monitor.check(telemetry)

    print(f"\n{report.summary()}")

    if report.is_ok:
        print("\n[OK] No significant drift detected. Model is safe to use.")
        return 0
    else:
        print("\n[WARNING] Drift detected! Consider retraining the model.")
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
