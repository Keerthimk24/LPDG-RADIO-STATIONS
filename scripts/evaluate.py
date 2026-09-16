#!/usr/bin/env python3
"""Evaluate predictions against cost metrics.

Usage:
    python scripts/evaluate.py --predictions predictions.csv --data ./data
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.loader import load_engineer_review, normalise_gateway_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate predictions against cost metrics")
    parser.add_argument("--predictions", type=str, default="predictions.csv",
                        help="Path to predictions CSV")
    parser.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "./data"),
                        help="Path to data directory")
    parser.add_argument("--cost-fp", type=float, default=380.0, help="Cost of false positive")
    parser.add_argument("--cost-fn", type=float, default=600.0, help="Cost of false negative per week")
    args = parser.parse_args(argv)

    import pathlib
    data_dir = pathlib.Path(args.data)

    # Load predictions
    preds = pd.read_csv(args.predictions)
    preds["gateway_id"] = preds["gateway_id"].apply(normalise_gateway_id)
    print(f"Predictions: {len(preds)} rows, {preds['week_start'].nunique()} weeks")

    # Load engineer review as ground truth (limited to Feb 2026 week)
    try:
        review = load_engineer_review(data_dir)
        bad_gateways = set(review[review["label"] == 1]["gateway_id"])
        print(f"Engineer review: {len(bad_gateways)} 'Schlecht' gateways")

        # For the Feb weeks, check how many bad gateways we caught
        feb_weeks = preds[preds["week_start"].str.startswith("2026-02")]
        for week, group in feb_weeks.groupby("week_start"):
            predicted_gws = set(group["gateway_id"])
            caught = predicted_gws & bad_gateways
            missed = bad_gateways - predicted_gws
            false_alarms = predicted_gws - bad_gateways

            week_cost = len(false_alarms) * args.cost_fp + len(missed) * args.cost_fn

            print(f"\n  Week {week}:")
            print(f"    Caught {len(caught)}/{len(bad_gateways)} bad gateways")
            print(f"    False alarms: {len(false_alarms)} (cost: €{len(false_alarms) * args.cost_fp:.0f})")
            print(f"    Missed: {len(missed)} (cost: €{len(missed) * args.cost_fn:.0f})")
            print(f"    Week cost: €{week_cost:.0f}")
    except Exception as e:
        print(f"Note: Could not load engineer review for evaluation: {e}")

    # Basic statistics
    print(f"\nPrediction statistics:")
    print(f"  Unique gateways across all weeks: {preds['gateway_id'].nunique()}")
    print(f"  Avg score: {preds['score'].mean():.4f}")
    print(f"  Score range: [{preds['score'].min():.4f}, {preds['score'].max():.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
