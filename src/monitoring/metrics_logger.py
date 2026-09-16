"""
Metrics logger — tracks performance metrics across model versions and runs.
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs and retrieves performance metrics."""

    def __init__(self, model_dir: pathlib.Path):
        self.metrics_path = model_dir / "metrics_log.json"
        self.model_dir = model_dir

    def _load_log(self) -> list[dict]:
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return []

    def _save_log(self, log: list[dict]):
        with open(self.metrics_path, "w") as f:
            json.dump(log, f, indent=2)

    def log_prediction_run(
        self,
        version: str,
        n_weeks: int,
        n_gateways_per_week: int,
        cost_fp: float,
        cost_fn: float,
        notes: str = "",
    ):
        """Log a prediction run."""
        log = self._load_log()
        log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": "prediction_run",
            "model_version": version,
            "n_weeks": n_weeks,
            "n_gateways_per_week": n_gateways_per_week,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "notes": notes,
        })
        self._save_log(log)
        logger.info("Logged prediction run: version=%s, %d weeks", version, n_weeks)

    def log_evaluation(
        self,
        version: str,
        total_cost: float,
        n_true_positives: int,
        n_false_positives: int,
        n_false_negatives: int,
        n_true_negatives: int,
        notes: str = "",
    ):
        """Log an evaluation result."""
        log = self._load_log()
        log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": "evaluation",
            "model_version": version,
            "total_cost_eur": total_cost,
            "true_positives": n_true_positives,
            "false_positives": n_false_positives,
            "false_negatives": n_false_negatives,
            "true_negatives": n_true_negatives,
            "precision": n_true_positives / max(n_true_positives + n_false_positives, 1),
            "recall": n_true_positives / max(n_true_positives + n_false_negatives, 1),
            "notes": notes,
        })
        self._save_log(log)
        logger.info("Logged evaluation: version=%s, cost=€%.0f", version, total_cost)

    def get_history(self, metric_type: str | None = None) -> list[dict]:
        """Retrieve metric history, optionally filtered by type."""
        log = self._load_log()
        if metric_type:
            return [entry for entry in log if entry.get("type") == metric_type]
        return log
