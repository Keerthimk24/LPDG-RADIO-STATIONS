"""Tests for model registry and rollback."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from src.monitoring.model_registry import ModelRegistry


def _create_mock_model(model_dir, version, cost=1000):
    """Helper to create a mock model version."""
    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save a simple "model" (just a dict for testing)
    model = {"type": "mock", "version": version}
    joblib.dump(model, version_dir / "model.joblib")

    # Save metadata
    metadata = {
        "version": version,
        "trained_on": "2026-01-01T00:00:00",
        "features": ["f1", "f2", "f3"],
        "cv_total_cost_eur": cost,
        "best_iteration": 100,
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return version_dir


class TestModelRegistry:
    def test_rollback_changes_version(self, tmp_model_dir):
        """Rolling back should change the active version."""
        registry = ModelRegistry(tmp_model_dir)

        # Create two versions
        _create_mock_model(tmp_model_dir, "v1", cost=1200)
        _create_mock_model(tmp_model_dir, "v2", cost=800)

        # Set current to v2
        (tmp_model_dir / "current_version.txt").write_text("v2")

        # Rollback to v1
        assert registry.get_current_version() == "v2"
        success = registry.rollback("v1", reason="v2 worse on new data")
        assert success
        assert registry.get_current_version() == "v1"

    def test_rollback_logs_event(self, tmp_model_dir):
        """Rollback should be logged in registry."""
        registry = ModelRegistry(tmp_model_dir)
        _create_mock_model(tmp_model_dir, "v1")
        _create_mock_model(tmp_model_dir, "v2")
        (tmp_model_dir / "current_version.txt").write_text("v2")

        registry.rollback("v1", reason="testing rollback")

        reg_data = registry._load_registry()
        assert len(reg_data.get("rollback_history", [])) > 0
        assert reg_data["rollback_history"][-1]["to_version"] == "v1"

    def test_rollback_to_nonexistent_fails(self, tmp_model_dir):
        """Rolling back to a version that doesn't exist should fail."""
        registry = ModelRegistry(tmp_model_dir)
        with pytest.raises(FileNotFoundError):
            registry.rollback("v999")

    def test_list_versions(self, tmp_model_dir):
        """Should list all registered versions."""
        registry = ModelRegistry(tmp_model_dir)
        _create_mock_model(tmp_model_dir, "v1")

        # Manually update registry
        reg_data = {"models": [{"version": "v1"}], "current": "v1"}
        with open(tmp_model_dir / "registry.json", "w") as f:
            json.dump(reg_data, f)

        versions = registry.list_versions()
        assert len(versions) >= 1

    def test_model_loads_after_rollback(self, tmp_model_dir):
        """Model should load correctly after rollback."""
        registry = ModelRegistry(tmp_model_dir)
        _create_mock_model(tmp_model_dir, "v1")
        _create_mock_model(tmp_model_dir, "v2")
        (tmp_model_dir / "current_version.txt").write_text("v2")

        registry.rollback("v1")

        # Verify model loads
        model = joblib.load(tmp_model_dir / "v1" / "model.joblib")
        assert model["version"] == "v1"
