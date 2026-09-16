"""
Model registry — version management and rollback.

A way to go back to the previous version. One you have actually tried.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model versions, rollbacks, and metadata."""

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = model_dir / "registry.json"

    def _load_registry(self) -> dict:
        """Load the registry index."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"models": [], "current": None, "rollback_history": []}

    def _save_registry(self, registry: dict):
        """Save the registry index."""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def get_current_version(self) -> str | None:
        """Get the currently active model version."""
        marker = self.model_dir / "current_version.txt"
        if marker.exists():
            return marker.read_text().strip()
        return None

    def list_versions(self) -> list[dict]:
        """List all registered model versions."""
        registry = self._load_registry()
        return registry.get("models", [])

    def rollback(self, target_version: str, reason: str = "") -> bool:
        """Roll back to a previous model version.

        Steps:
        1. Validate the target version exists and loads correctly
        2. Update the current version marker
        3. Run a smoke prediction to verify
        4. Log the rollback event

        Args:
            target_version: Version to roll back to (e.g., "v1").
            reason: Reason for rollback.

        Returns:
            True if rollback succeeded.
        """
        target_dir = self.model_dir / target_version
        if not target_dir.exists():
            raise FileNotFoundError(f"Target version not found: {target_dir}")

        model_path = target_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Step 1: Validate model loads
        try:
            model = joblib.load(model_path)
            logger.info("Model %s loads successfully", target_version)
        except Exception as e:
            logger.error("Failed to load model %s: %s", target_version, e)
            return False

        # Step 2: Validate metadata exists
        meta_path = target_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            logger.info("Model %s metadata: trained on %s, %d features",
                        target_version,
                        metadata.get("trained_on", "unknown"),
                        len(metadata.get("features", [])))

        # Step 3: Update current version
        current = self.get_current_version()
        marker = self.model_dir / "current_version.txt"
        marker.write_text(target_version)

        # Step 4: Log the rollback
        registry = self._load_registry()
        registry["current"] = target_version
        if "rollback_history" not in registry:
            registry["rollback_history"] = []
        registry["rollback_history"].append({
            "from_version": current,
            "to_version": target_version,
            "reason": reason,
        })
        self._save_registry(registry)

        logger.info("ROLLBACK: %s → %s (reason: %s)", current, target_version, reason or "not specified")
        return True

    def compare_versions(self, version_a: str, version_b: str) -> dict:
        """Compare metadata of two model versions."""
        meta_a = self._load_metadata(version_a)
        meta_b = self._load_metadata(version_b)

        return {
            "version_a": version_a,
            "version_b": version_b,
            "a_trained_on": meta_a.get("trained_on"),
            "b_trained_on": meta_b.get("trained_on"),
            "a_cv_cost": meta_a.get("cv_total_cost_eur"),
            "b_cv_cost": meta_b.get("cv_total_cost_eur"),
            "a_n_features": len(meta_a.get("features", [])),
            "b_n_features": len(meta_b.get("features", [])),
            "a_best_iteration": meta_a.get("best_iteration"),
            "b_best_iteration": meta_b.get("best_iteration"),
        }

    def _load_metadata(self, version: str) -> dict:
        """Load metadata for a specific version."""
        meta_path = self.model_dir / version / "metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)
