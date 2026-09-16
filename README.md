# LPDG Gateway Health Predictor

**Which 15 gateways should our field team visit next week?**

A cost-optimised ML system that predicts which IoT gateways need field visits, ranking them by expected cost savings. Built for the LPDG Innovation Hub Selection Challenge 2026.

**Areas covered:** Machine Learning (primary) · Data Science · MLOps

## 📹 Screen Recording

> **[→ Watch the 6–8 minute walkthrough (TODO: add link before submission)]()**
>
> Covers: single-command run, how the model ranks gateways, cost comparison vs baseline, and the live-session rollback demo.

---

## Quick Start

### With Docker (recommended — one command)

```bash
# Train + predict + output predictions.csv — all in one command
docker compose up --build
```

This auto-trains a model if none exists, generates `predictions.csv`, and validates it. To retrain explicitly or run tests:

```bash
docker compose --profile train up --build   # Force retrain
docker compose --profile test up --build    # Run 50+ tests
```

### Without Docker

```bash
pip install -r requirements.txt

# Single command: auto-trains if needed, then predicts
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

### What the Model Uses, Ignores, and Pays Attention To

**What was fed (~130 feature columns across 7 groups — 68 unique feature concepts, expanding via one-hot encoding):**
- 7-day and 28-day window aggregations of telemetry: mean/max/std of `offline_duration_sec`, `disconnection_cnt`, `reboot_cnt`, signal quality (RSSI/RSCP/ECIO good/normal/bad ratios), network type (2G/3G/4G mix), memory and CPU load
- Trend ratios (7d ÷ 28d) for all telemetry metrics — a ratio > 1 means degradation
- Meter read success rate (latest week + 4-week trend), meters at risk count
- Gateway metadata: hardware model, firmware version, antenna type, age since installation, number of meters
- Visit history: past fault rate, repeat offender flag, days since last visit

**What was deliberately left out:**
- Raw hourly time series (too noisy — the model sees aggregated statistics, not 168 hourly values)
- Self-resolving cellular blips (filtered out by using 7-day aggregations rather than single-hour spikes)
- Gateway ID itself (no memorising — the model cannot learn "gateway X is always bad")
- Future data: strict temporal cutoff ensures features for Monday only use data from before that Monday

**What the model pays attention to most (top drivers by LightGBM gain):**
1. `meter_read_rate_latest` — direct proxy for "is this gateway delivering value?"
2. `trend_offline_duration_sec_ratio` — is offline time getting worse?
3. `meters_at_risk` — economic impact (more meters = higher cost of failure)
4. `7d_disconnection_cnt_mean` — frequency of network drops in the recent week
5. `visit_fault_rate` — repeat offenders: gateways where past visits found real faults

### Cost Comparison: ML Model vs 3-Sigma Baseline

Evaluated on the February 2026 scored weeks using the engineer review as ground truth (60 "Schlecht" gateways):

| Metric | 3-Sigma Baseline | ML Cost-Optimised |
|---|---|---|
| Bad gateways caught (avg/week) | 3 out of 60 | 8 out of 60 |
| False alarms (avg/week) | 12 | 7 |
| Weekly cost (avg) | €38,760 | €33,860 |
| **Weekly savings vs baseline** | — | **€4,900/week** |
| **Annualised savings** | — | **~€255,000/year** |

The model catches **2.7× more broken gateways** while sending **42% fewer false alarms**.

### What Happens When the Network Changes

**New gateways added to the fleet:**
- New gateways automatically get features from `gateway_master` and telemetry. Visit history features default to 0 (no history), so the model relies on telemetry and meter reads. Tested in `test_generalization.py::TestUnseenGateways`.

**Firmware updates or hardware swaps:**
- The `fw_version` and `hw_model` features are categorical. A new firmware version the model hasn't seen will get a default encoding. If the update changes telemetry patterns significantly, the drift detector (KS-test on core metrics) will flag it.

**Drift detection and response:**
- `DataDriftMonitor` runs a two-sample Kolmogorov-Smirnov test on each core metric (`offline_duration_sec`, `disconnection_cnt`, `reboot_cnt`) comparing current data to the training data distribution. If p < 0.05 on ≥2 metrics for ≥2 weeks, the system logs a warning and retraining is recommended.
- Rollback is a one-command operation: `python scripts/rollback.py --to v1`.

---

## Project Structure

```
src/
├── config.py                  # All settings from environment variables + YAML
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
│   ├── trainer.py             # Training pipeline (GroupKFold CV)
│   ├── predictor.py           # Inference (separate from training)
│   ├── ranker.py              # Score → top-15 with reasons
│   └── explainer.py           # SHAP-based explanations
├── monitoring/
│   ├── data_drift.py          # Schema/distribution drift detection
│   ├── model_registry.py      # Versioning + rollback
│   └── metrics_logger.py      # Performance tracking
├── dashboard/
│   └── report.py              # Interactive HTML dashboard generator
└── utils/
    ├── gateway_ids.py         # ID normalisation (hex ↔ colon)
    └── logging_setup.py       # Structured logging
```

---

## Configuration

All settings come from environment variables, with optional YAML defaults in `config/default.yaml`:

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

Copy `.env.example` to `.env` and modify as needed. All hyperparameters and model settings are also documented in `config/default.yaml`.

### Available Commands

| Command | What It Does |
|---|---|
| `make all` | Train → Predict → Validate (default) |
| `make train` | Train the LightGBM model |
| `make predict` | Generate predictions.csv + validate |
| `make validate` | Check predictions.csv format |
| `make evaluate` | Cost comparison vs ground truth |
| `make dashboard` | Generate interactive HTML dashboard |
| `make test` | Run all tests |
| `make drift` | Data drift detection report |
| `make rollback VERSION=v1` | Rollback to a previous model version |
| `make clean` | Remove generated files |

---

## How to Tell It Is Working

1. **Training succeeds**: `scripts/train.py` prints "Training complete → models/v1" and creates model files
2. **Predictions validate**: `validate_submission.py predictions.csv` prints "OK"
3. **Tests pass**: `pytest tests/ -v` → 55 tests pass across 7 suites
4. **Dashboard**: `make dashboard` generates `dashboard.html` with interactive charts
5. **Logs**: Set `LOG_LEVEL=DEBUG` for detailed output; `INFO` for normal operation

## What To Do When It Is Not Working

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: DATA_DIR` | Data not mounted | Check `DATA_DIR` env var, ensure `./data/` exists |
| `FileNotFoundError: No current model` | Auto-training failed | Check training logs for data issues; run `docker compose --profile train up` manually |
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
- A/B test framework to compare model versions on live data
- Real-time streaming alerting (vs current weekly batch predictions)
