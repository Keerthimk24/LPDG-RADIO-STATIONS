"""
Data loader — reads all data sources from the data directory.

Handles both parquet (partitioned telemetry) and CSV/XLSX files.
Gateway IDs are normalised to 12-char uppercase hex (no colons) everywhere.
"""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
import re

import pandas as pd

logger = logging.getLogger(__name__)

_COLON_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def normalise_gateway_id(gw_id: str) -> str:
    """Convert any gateway ID format to 12-char uppercase hex (no colons)."""
    text = str(gw_id).strip()
    if _COLON_PATTERN.match(text):
        return text.replace(":", "").upper()
    # Already bare hex or close to it
    return re.sub(r"[^0-9A-Fa-f]", "", text).upper()


def load_telemetry(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load the full telemetry dataset from partitioned parquet files.

    Returns a DataFrame with datetime index `ts` and normalised `gateway_id`.
    """
    telemetry_dir = data_dir / "telemetry"
    if not telemetry_dir.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_dir}")

    logger.info("Loading telemetry from %s", telemetry_dir)
    frame = pd.read_parquet(telemetry_dir)
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    frame["gateway_id"] = frame["gateway_id"].apply(normalise_gateway_id)
    frame = frame.drop(columns=["ts_utc"])
    logger.info("Loaded telemetry: %d rows, %d gateways, %s to %s",
                len(frame), frame["gateway_id"].nunique(),
                frame["ts"].min().date(), frame["ts"].max().date())
    return frame


def load_gateway_master(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load the gateway asset register."""
    path = data_dir / "gateway_master.csv"
    logger.info("Loading gateway master from %s", path)
    frame = pd.read_csv(path, encoding="latin-1")
    frame["gateway_id"] = frame["gateway_id"].apply(normalise_gateway_id)

    # Parse dates
    for col in ["fw_updated_on", "installed_on", "decommissioned_on"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")

    logger.info("Loaded gateway master: %d gateways, %d active",
                len(frame), frame["decommissioned_on"].isna().sum())
    return frame


def load_field_visits(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load historical field visit work orders."""
    path = data_dir / "field_visits.csv"
    logger.info("Loading field visits from %s", path)
    frame = pd.read_csv(path)
    frame["gateway_id"] = frame["gateway_id"].apply(normalise_gateway_id)

    for col in ["requested_on", "visited_on"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")

    logger.info("Loaded field visits: %d visits, %d unique gateways",
                len(frame), frame["gateway_id"].nunique())
    return frame


def load_meter_read_success(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load meter read success rates (weekly, per gateway)."""
    path = data_dir / "meter_read_success.csv"
    logger.info("Loading meter read success from %s", path)
    frame = pd.read_csv(path)
    frame["gateway_id"] = frame["gateway_id"].apply(normalise_gateway_id)
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame["read_success_rate"] = (
        frame["meters_read"] / frame["meters_expected"].replace(0, 1)
    ).clip(0, 1)

    logger.info("Loaded meter read success: %d rows, %d weeks",
                len(frame), frame["week_start"].nunique())
    return frame


def load_engineer_review(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load engineer review labels (ground truth for supervised learning)."""
    path = data_dir / "engineer_review_2026-02.xlsx"
    logger.info("Loading engineer review from %s", path)
    frame = pd.read_excel(path)
    frame["gateway_id"] = frame["gateway_id"].apply(normalise_gateway_id)

    # Binary label: 1 = Schlecht (bad), 0 = Normal
    frame["label"] = (frame["Kategorie"] == "Schlecht").astype(int)

    logger.info("Loaded engineer review: %d gateways (%d Schlecht, %d Normal)",
                len(frame), frame["label"].sum(), (1 - frame["label"]).sum())
    return frame


def load_all(data_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Load all datasets and return as a dictionary."""
    return {
        "telemetry": load_telemetry(data_dir),
        "gateway_master": load_gateway_master(data_dir),
        "field_visits": load_field_visits(data_dir),
        "meter_read_success": load_meter_read_success(data_dir),
        "engineer_review": load_engineer_review(data_dir),
    }
