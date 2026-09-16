# LPDG Gateway Health Predictor

**Which 15 gateways should our field team visit next week?**

A cost-optimised ML system that predicts which IoT gateways need field visits, ranking them by expected cost savings. Built for the LPDG Innovation Hub Selection Challenge 2026.

**Areas covered:** Machine Learning (primary) · Data Science · MLOps

---

## Quick Start

### With Docker (recommended — one command)

```bash
# 1. Train the model
docker compose --profile train up --build

# 2. Generate predictions
docker compose up --build

# 3. Validate
python validate_submission.py predictions.csv
```

### Without Docker

```bash
pip install -r requirements.txt

# Train
python scripts/train.py --data ./data --model-dir ./models

# Predict
python scripts/predict.py --data ./data --out predictions.csv

# Validate
python validate_submission.py predictions.csv
```

### Run Tests

```bash
# Docker
docker compose --profile test up --build

# Local
pytest tests/ -v
```

---

## How It Works

### The Problem
~320 gateways relay meter readings. When one fails silently, meters stop being read. The operations team gets 15 visits/week. Today they pick from a spreadsheet and gut feel — 61% of visits find nothing wrong.

### The Approach

1. **Feature Engineering** — For each Monday, compute features from:
   - **Telemetry** (7d & 28d windows): offline duration, disconnections, reboots, signal quality, trends
   - **Meter reads**: success rate, meters at risk, declining trend
   - **Gateway metadata**: hardware model, firmware, age, number of meters
   - **Visit history**: past fault rate, repeat offender status, days since last visit

2. **Model** — LightGBM with cost-sensitive objective (`scale_pos_weight = 600/380`):
   - False negative (miss a broken gateway): €600/week
   - False positive (unnecessary visit): €380
   - Optimises total cost, not accuracy

3. **Ranking** — For each week, rank all gateways by `P(broken) × 600 - (1-P(broken)) × 380`, take top 15

4. **Reasons** — Each ranked gateway gets a human-readable reason (≤300 chars) for the operations manager

---

## Project Structure

```
src/
├── config.py                  # All settings from environment variables
├── pipeline.py                # End-to-end train & predict orchestrator
├── data/
│   ├── loader.py              # Data loading (parquet, CSV, XLSX)
│   └── validator.py           # Schema + quality validation
├── features/
│   ├── telemetry_features.py  # 7d/28d window aggregations + trends
│   ├── meter_features.py      # Meter read success features
│   ├── gateway_features.py    # Static metadata features
│   ├── visit_features.py      # Visit history features
│   └── builder.py             # Feature pipeline orchestrator
├── model/
│   ├── cost_function.py       # Asymmetric cost logic + ranking
│   ├── trainer.py             # Training pipeline
│   ├── predictor.py           # Inference (separate from training)
│   ├── ranker.py              # Score → top-15 with reasons
│   └── explainer.py           # SHAP-based explanations
└── monitoring/
    ├── data_drift.py          # Schema/distribution drift detection
    ├── model_registry.py      # Versioning + rollback
    └── metrics_logger.py      # Performance tracking
```

---

## Configuration

All settings come from environment variables (no editing files):

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Path to data directory |
| `MODEL_DIR` | `./models` | Path to model artifacts |
| `PREDICTIONS_OUT` | `./predictions.csv` | Output file path |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `VISIT_BUDGET` | `15` | Visits per week |
| `COST_FALSE_POSITIVE` | `380` | Cost of unnecessary visit (€) |
| `COST_FALSE_NEGATIVE` | `600` | Cost/week of missed broken gateway (€) |
| `MODEL_VERSION` | `current` | Which model version to use |

Copy `.env.example` to `.env` and modify as needed.

---

## How to Tell It Is Working

1. **Training succeeds**: `scripts/train.py` prints "Training complete → models/v1" and creates model files
2. **Predictions validate**: `validate_submission.py predictions.csv` prints "OK"
3. **Tests pass**: `pytest tests/ -v` → 43 tests pass
4. **Health check**: Docker container health check runs `Config.validate()` every 30 seconds
5. **Logs**: Set `LOG_LEVEL=DEBUG` for detailed output; `INFO` for normal operation

## What To Do When It Is Not Working

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: DATA_DIR` | Data not mounted | Check `DATA_DIR` env var, ensure `./data/` exists |
| `FileNotFoundError: No current model` | Haven't trained yet | Run `docker compose --profile train up` first |
| Training produces all zeros | Imbalanced labels | Check label distribution in logs; adjust `scale_pos_weight` |
| Predictions don't validate | Wrong week range or count | Check `scored_weeks` in config matches the 8 Mondays |
| Drift detected warning | New data has different distribution | Check if data format changed; consider retraining |
| Model rollback needed | New version performs worse | `python scripts/rollback.py --to v1` |

---

## MLOps: Model Versioning & Rollback

### Model Registry

Each trained model is saved to `models/v{N}/` with:
- `model.joblib` — serialized model
- `metadata.json` — training params, features, metrics, git SHA, timestamp
- `feature_schema.json` — expected input columns
- `drift_reference.json` — training data statistics for drift detection

### Rollback

```bash
# Roll back to a previous version
python scripts/rollback.py --to v1 --reason "v2 worse on new data"

# Check current version
cat models/current_version.txt
```

### Retraining Policy

**When to retrain:**
1. Drift detector flags KS-test p < 0.05 on ≥2 core metrics for ≥2 weeks
2. New month of data available (monthly cadence)
3. Operations manager reports model "getting it wrong"

**How to know retraining made things worse:**
1. Compare old and new model on holdout validation set
2. New model's total cost must be ≤ old model's cost
3. If worse → keep old model, log the reason

---

## What It Cannot Do

1. **Cannot predict sudden hardware failures** — the model learns gradual degradation patterns, not one-off catastrophic events
2. **Cannot account for external events** — power grid outages, severe weather, or network provider changes are not in the training data
3. **Limited by label quality** — we constructed labels from engineer review (120 gateways) + visit outcomes (61% false positive rate) + meter reads. The "true" set of broken gateways is unknown
4. **Assumes static visit budget** — if the team could do 20 visits/week, the optimal threshold would shift. The model ranks by expected cost but is calibrated for 15 visits
5. **No real-time monitoring** — this is a batch prediction system (weekly), not a streaming alerting system

**What another two weeks would fix:**
- Hyperparameter tuning with Optuna (systematic search vs manual selection)
- Ensemble with baseline 3-sigma (hedge bets between ML and rule-based)
- Interactive dashboard for the operations manager (Streamlit or Dash)
- A/B test framework to compare model versions on live data
