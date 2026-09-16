"""Tests for multi-model training, prediction, and serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.model.predictor import predict
from src.model.ranker import rank_predictions
from src.model.trainer import SUPPORTED_MODELS, save_model, train_model


@pytest.fixture
def synthetic_data():
    """Create minimal synthetic dataset for fast model testing."""
    np.random.seed(42)
    n_samples = 100
    features = {
        f"feat_{i}": np.random.normal(0, 1, n_samples)
        for i in range(10)
    }
    # Introduce some NaN values to test imputer / native NaN handling
    features["feat_0"][::10] = np.nan
    features["feat_1"][::15] = np.nan

    X = pd.DataFrame(features, index=[f"GW_{i:03d}" for i in range(n_samples)])
    X["_monday"] = "2026-02-02"

    # Synthetic binary label with signal
    y_prob = 1 / (1 + np.exp(-features["feat_2"] + 0.5 * np.nan_to_num(features["feat_0"])))
    y = pd.Series((y_prob > 0.5).astype(float), index=X.index, name="label")

    return X, y


class TestMultiModelTraining:
    @pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
    def test_train_and_predict(self, synthetic_data, tmp_path, model_type):
        """Every supported model type should train, predict, and generate valid probabilities."""
        X, y = synthetic_data
        config = Config(model_dir=tmp_path, random_seed=42)

        feat_cols = [c for c in X.columns if c not in ("_monday", "label")]
        model, metadata = train_model(X, y, config, feature_columns=feat_cols, model_type=model_type)

        assert metadata["model_type"] == model_type
        assert "cv_total_cost_eur" in metadata

        # Generate predictions
        preds = predict(model, X, metadata)
        assert len(preds) == len(X)
        assert "probability" in preds.columns
        assert (preds["probability"] >= 0.0).all()
        assert (preds["probability"] <= 1.0).all()

        # Rank predictions
        ranked = rank_predictions(preds, X, budget=10)
        assert len(ranked) == 10
        assert list(ranked["rank"]) == list(range(1, 11))

    def test_invalid_model_type_raises(self, synthetic_data, tmp_path):
        """An unsupported model type should raise a ValueError."""
        X, y = synthetic_data
        config = Config(model_dir=tmp_path)
        feat_cols = [c for c in X.columns if c not in ("_monday", "label")]

        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X, y, config, feature_columns=feat_cols, model_type="quantum_net")

    def test_save_and_reload_sklearn_model(self, synthetic_data, tmp_path):
        """Saved sklearn models should reload and produce identical predictions."""
        X, y = synthetic_data
        config = Config(model_dir=tmp_path, random_seed=42)
        feat_cols = [c for c in X.columns if c not in ("_monday", "label")]

        model, metadata = train_model(X, y, config, feature_columns=feat_cols, model_type="random_forest")
        version_dir = save_model(model, metadata, tmp_path, version="v_test_rf")

        from src.model.predictor import load_model
        loaded_model, loaded_meta = load_model(tmp_path, version="v_test_rf")

        preds1 = predict(model, X, metadata)
        preds2 = predict(loaded_model, X, loaded_meta)

        pd.testing.assert_series_equal(preds1["probability"], preds2["probability"])
