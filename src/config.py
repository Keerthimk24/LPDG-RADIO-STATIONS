"""
Configuration management — all settings come from environment variables.

No paths that only exist on your laptop. No editing files to change settings.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""

    # Paths
    data_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(os.getenv("DATA_DIR", "./data")))
    model_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(os.getenv("MODEL_DIR", "./models")))
    predictions_out: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(os.getenv("PREDICTIONS_OUT", "./predictions.csv"))
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Reproducibility
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))

    # Business constraints
    visit_budget: int = field(default_factory=lambda: int(os.getenv("VISIT_BUDGET", "15")))
    cost_fp: float = field(default_factory=lambda: float(os.getenv("COST_FALSE_POSITIVE", "380")))
    cost_fn: float = field(default_factory=lambda: float(os.getenv("COST_FALSE_NEGATIVE", "600")))

    # Drift detection
    drift_threshold: float = field(default_factory=lambda: float(os.getenv("DRIFT_THRESHOLD", "0.05")))

    # Model version & architecture
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "current"))
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "lightgbm"))

    # Scored weeks: 8 Mondays from 2 Feb to 23 Mar 2026
    scored_weeks: tuple = field(default=(
        "2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23",
        "2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23",
    ))

    # Feature engineering windows
    baseline_days: int = 28
    recent_days: int = 7

    def __post_init__(self):
        """Set up logging on creation."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @classmethod
    def validate(cls) -> bool:
        """Health check: can we load config and find data?"""
        cfg = cls()
        if not cfg.data_dir.exists():
            raise FileNotFoundError(f"DATA_DIR does not exist: {cfg.data_dir}")
        telemetry_dir = cfg.data_dir / "telemetry"
        if not telemetry_dir.exists():
            raise FileNotFoundError(f"telemetry directory not found: {telemetry_dir}")
        required_files = ["gateway_master.csv", "field_visits.csv", "meter_read_success.csv"]
        for fname in required_files:
            if not (cfg.data_dir / fname).exists():
                raise FileNotFoundError(f"Required file missing: {cfg.data_dir / fname}")
        logger.info("Config validated OK: data_dir=%s, model_dir=%s", cfg.data_dir, cfg.model_dir)
        return True
