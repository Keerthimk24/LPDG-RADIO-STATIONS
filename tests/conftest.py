"""Shared test fixtures."""

from __future__ import annotations

import datetime as dt
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_telemetry():
    """Create a small synthetic telemetry DataFrame for testing."""
    np.random.seed(42)
    n_gateways = 10
    n_hours = 24 * 30  # One month

    gateways = [f"GW{i:010X}00" for i in range(n_gateways)]
    rows = []
    for gw in gateways:
        for h in range(n_hours):
            ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=h)
            rows.append({
                "gateway_id": gw,
                "ts": ts,
                "DateDt": ts.strftime("%Y-%m-%d"),
                "hour": ts.hour,
                "offline_duration_sec": max(0, int(np.random.exponential(50))),
                "disconnection_cnt": int(np.random.poisson(0.3)),
                "reboot_cnt": int(np.random.poisson(0.05)),
                "reboot_duration_sec": int(np.random.exponential(10)),
                "reboot_importance": np.random.uniform(0, 1),
                "no_conn_importance": np.random.uniform(0, 1),
                "online_duration_mins": max(0, 60 - np.random.exponential(5)),
                "avg_load1": np.random.uniform(0.1, 2.0),
                "avg_memfree": np.random.uniform(10000, 50000),
                "rx_nr_pkts": int(np.random.exponential(30)),
                "rssi_good": int(np.random.randint(0, 14)),
                "rssi_normal": int(np.random.randint(0, 14)),
                "rssi_bad": int(np.random.randint(0, 5)),
                "rscp_rsrp_good": int(np.random.randint(0, 14)),
                "rscp_rsrp_normal": int(np.random.randint(0, 14)),
                "rscp_rsrp_bad": int(np.random.randint(0, 5)),
                "ecio_rsrq_good": int(np.random.randint(0, 14)),
                "ecio_rsrq_normal": int(np.random.randint(0, 14)),
                "ecio_rsrq_bad": int(np.random.randint(0, 5)),
                "network_2g": int(np.random.randint(0, 5)),
                "network_3g": int(np.random.randint(0, 5)),
                "network_4g": int(np.random.randint(0, 14)),
                "network_unknown": 0,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_gateway_master():
    """Create a small synthetic gateway master DataFrame."""
    gateways = [f"GW{i:010X}00" for i in range(10)]
    return pd.DataFrame({
        "gateway_id": gateways,
        "tenant": ["tenant_a"] * 6 + ["tenant_b"] * 4,
        "site_type": ["Gebaeude"] * 5 + ["Heizraum"] * 5,
        "region": ["Bayern"] * 3 + ["Hessen"] * 4 + ["Sachsen"] * 3,
        "hw_model": ["GW-2100"] * 6 + ["GW-2100L"] * 4,
        "antenna_type": ["Omni 3dBi"] * 5 + ["Omni 5dBi"] * 5,
        "fw_version": ["3.2.0"] * 4 + ["2.15.1"] * 3 + ["3.3.1"] * 3,
        "fw_updated_on": pd.to_datetime(["2025-06-01"] * 10),
        "installed_on": pd.to_datetime(["2024-01-15"] * 10),
        "decommissioned_on": [pd.NaT] * 10,
        "n_meters_installed": [150, 200, 100, 300, 80, 120, 250, 180, 90, 160],
    })


@pytest.fixture
def sample_meter_reads():
    """Create synthetic meter read success data."""
    gateways = [f"GW{i:010X}00" for i in range(10)]
    rows = []
    for week_offset in range(8):
        week_start = dt.date(2026, 1, 5) + dt.timedelta(weeks=week_offset)
        for gw in gateways:
            expected = np.random.randint(50, 300)
            read_rate = np.random.uniform(0.6, 1.0)
            rows.append({
                "gateway_id": gw,
                "week_start": pd.Timestamp(week_start),
                "meters_expected": expected,
                "meters_read": int(expected * read_rate),
                "read_success_rate": read_rate,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_field_visits():
    """Create synthetic field visit data."""
    gateways = [f"GW{i:010X}00" for i in range(5)]
    return pd.DataFrame({
        "visit_id": [f"WO-{i}" for i in range(10)],
        "gateway_id": gateways * 2,
        "requested_on": pd.to_datetime(["2025-10-01"] * 10),
        "visited_on": pd.to_datetime(["2025-10-05"] * 10),
        "reason_reported": ["Keine Verbindung"] * 4 + ["Routinepruefung"] * 6,
        "outcome": ["Fehler behoben"] * 4 + ["Kein Fehler gefunden"] * 6,
        "parts_replaced": [None] * 6 + ["Antenne"] * 4,
        "technician_hours": np.random.uniform(0.5, 3.0, 10),
    })


@pytest.fixture
def sample_engineer_review():
    """Create synthetic engineer review data."""
    gateways = [f"GW{i:010X}00" for i in range(10)]
    return pd.DataFrame({
        "gateway_id": gateways,
        "standort": ["Bayern / Gebaeude"] * 10,
        "Kategorie": ["Schlecht"] * 4 + ["Normal"] * 6,
        "reviewed_on": pd.to_datetime(["2026-02-15"] * 10),
        "reviewer": ["M. Hoffmann"] * 10,
        "Bemerkung": [None] * 10,
        "label": [1] * 4 + [0] * 6,
    })


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
