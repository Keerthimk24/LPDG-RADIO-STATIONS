#!/usr/bin/env python3
"""Roll back to a previous model version.

Usage:
    python scripts/rollback.py --to v1 --reason "v2 performed worse on new data"
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.model_registry import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll back to a previous model version")
    parser.add_argument("--to", required=True, type=str, help="Target version (e.g., v1)")
    parser.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", "./models"),
                        help="Path to model artifacts")
    parser.add_argument("--reason", type=str, default="", help="Reason for rollback")
    args = parser.parse_args(argv)

    model_dir = pathlib.Path(args.model_dir)
    registry = ModelRegistry(model_dir)

    current = registry.get_current_version()
    print(f"Current model version: {current}")
    print(f"Rolling back to: {args.to}")

    try:
        success = registry.rollback(args.to, reason=args.reason)
        if success:
            print(f"\nRollback successful: {current} → {args.to}")
            new_current = registry.get_current_version()
            print(f"Active version is now: {new_current}")
            return 0
        else:
            print("Rollback failed.", file=sys.stderr)
            return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
