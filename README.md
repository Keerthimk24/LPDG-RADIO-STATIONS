# 🏗️ LPDG — Gateway Health Prediction

> **AI-powered predictive maintenance for LoRaWAN gateways — Predicting which of 320 gateways need engineer visits each week**

[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)]()
[![License](https://img.shields.io/badge/license-Challenge%20Submission-lightgrey.svg)]()

---

## 📹 6–8 Minute Video Walkthrough & Resume

| Item | Link / Location | Description |
| :--- | :--- | :--- |
| 🎥 **Video Walkthrough (6–8 min)** | **[Watch on Google Drive](https://drive.google.com/file/d/1wDi-DuQ4l0REjnqG_6AtXIik7fPECC9h/view?usp=drive_link)** | Full screen recording covering: pipeline run, ML cost optimization, live dashboard, and rollback |
| 📄 **Candidate Resume** | **[View Resume (Google Drive)](https://drive.google.com/file/d/1ca27oPpk9432qBHjfyQgobw-EZOWjhtt/view?usp=drive_link)** · **[PDF](23091A3259_resume.pdf)** | Profile of **keerthi machanooru** ([@Keerthimk24](https://github.com/Keerthimk24)) |
| 📝 **Video Presentation Script** | **[View Presentation Script (Markdown)](PRESENTATION_SCRIPT.md)** · **[ElevenLabs Script](ELEVENLABS_SCRIPT.txt)** | Minute-by-minute speaking guide & verified SSML pause tags for the 6:55 evaluation video |

---

## 📊 Results at a Glance

| Metric | 3σ Baseline | Our Model | Delta |
| :--- | :---: | :---: | :---: |
| **Bad gateways caught (2-week window)** | 6 | 10 | **+67%** ✅ |
| **Bad gateways missed (2-week window)** | 54 | 50 | **−4** |
| **Total cost (2-week window)** | €79,800 | €77,400 | **−€2,400** |
| **Total cost (4-week February)** | €156,020 | €125,640 | **−€30,380** |
| **AUC-ROC** | — | **0.888** | — |
| **Training time** | — | **~10 seconds** | — |

> **Bottom line:** Our model catches **67% more broken gateways** than the statistical baseline, saving **€30,380** across the February evaluation period — while strictly adhering to the **15 visits/week** operational constraint.

---

## 🚀 Quick Start (2 minutes)

### Prerequisites
- **Docker** (recommended) **OR** **Python 3.12+**
- Challenge data file (`03-challenge-data.zip`)

### Step 1 — Set Up Data
```bash
git clone https://github.com/Keerthimk24/LPDG-RADIO-STATIONS.git
cd LPDG-RADIO-STATIONS

# Extract challenge data into data/ folder
unzip 03-challenge-data.zip -d data/
```

Your `data/` folder should look like this:
```
data/
├── telemetry/                      # Parquet files — hourly gateway metrics
├── gateway_master.csv              # 332 gateways (metadata)
├── field_visits.csv                # 642 historical visit outcomes
├── meter_read_success.csv          # Weekly meter read success rates
└── engineer_review_2026-02.xlsx    # Ground truth (60 Schlecht / 60 Normal)
```

### Step 2 — Run the Pipeline

#### Option A — Docker (one command, zero setup):
```bash
# Default: runs prediction pipeline and outputs predictions.csv
docker compose up
```

#### Option B — Local Python:
```bash
pip install -r requirements.txt

# On Windows (PowerShell / Command Prompt):
python scripts/predict.py --data data --out predictions.csv
python validate_submission.py predictions.csv

# On macOS / Linux (Terminal with make):
make all
```

**What happens:**
1. `→ Training model...` ✓ **~10s** — LightGBM booster with cost-sensitive weighting
2. `→ Generating predictions...` ✓ **~5s** — 120 rows (8 weeks × 15 gateways)
3. `→ Validating predictions...` ✓ `predictions.csv: OK`
4. `✓ Pipeline complete.` `predictions.csv` is ready for grading.

### Step 3 — The 2 Universal Commands (Works on EVERY Computer)

You only need **2 simple Python commands** to inspect everything:

#### 1️⃣ Command 1 — View Results, Accuracy, AUC-ROC & Cost Savings:
```bash
python scripts/evaluate.py
```
> **What it does:** Displays the complete executive evaluation report in your terminal:
> • **AUC-ROC: 0.888** | Recall@15: **67.0%** | Precision@15: **68.3%**
> • **Cost Comparison:** Saves **€30,380** over baseline in February (+67% more broken gateways caught)
> • **Week-by-week ground truth validation** against the 60 "Schlecht" reviewed gateways.

#### 2️⃣ Command 2 — Launch & View the Interactive Dashboard:
```bash
python scripts/dashboard.py
```
> **What it does:** Automatically builds the latest executive report (`dashboard.html`) and opens it directly in your web browser:
> • **Fleet Health Cards:** 302 Safe vs 30 Broken gateways
> • **4 Intuitive Charts:** Weekly Cost, Risk Tiers, Root Causes, Cumulative Savings
> • **Interactive Schedule:** Filterable weekly table with genuine risk out of 100, diagnosed fault reason, and recommended engineer action.

---

### Alternative / Advanced Commands:
```bash
# Generate predictions directly:
python scripts/predict.py --data data --out predictions.csv

# Validate predictions file format:
python validate_submission.py predictions.csv

# Run multi-model benchmark (LightGBM vs RandomForest vs others):
python scripts/compare_models.py

# Run all 55 tests:
pytest tests -q
```

---

## 🎯 Problem Statement

A utility company operates ~320 LoRaWAN gateways that relay smart meter readings. When a gateway fails silently, meter data stops flowing — costing **€600/week** in penalties per undetected failure.

- **The constraint:** Only **15 engineer visits** can be scheduled per week.
- **The challenge:** Each week, rank and pick the 15 gateways most likely to need engineer visits.

| Operational Outcome | Financial Impact |
| :--- | :--- |
| **✅ Visit a truly broken gateway** | **€380** (visit cost) — but saves **€600/week** in ongoing penalties (Net **+€220**) |
| **❌ Visit a healthy gateway** | **€380** wasted |
| **❌ Miss a broken gateway** | **€600/week** penalty continues |

**Our approach:** Train a LightGBM classifier with an asymmetric loss function derived from financial breakeven analysis ($P \approx 0.388$) on ~130 engineered features, selecting and explaining the top 15 each week.

---

## 🏗️ How It Works — Data Flow

```text
RAW DATA
┌────────────────────────────────────────────────────────┐
│ 📡 Telemetry: 1.4M rows (hourly metrics in Parquet)    │
│ 🏭 Gateway Master: 332 gateways (hardware & metadata)   │
│ 🔧 Field Visits: 642 visits (outcomes & technician notes)│
│ 📊 Meter Reads: 7,226 rows (weekly success percentages)│
│ 📋 Engineer Review: 120 labels (ground truth audit)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ DATA LOADING & PREPROCESSING                           │
│ • ID normalization (bare hex ↔ colon-separated)        │
│ • UTF-8 / Latin-1 encoding resolution                  │
│ • Strict schema & data type validation                 │
│ • Decommissioned gateway removal per week              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ ~130 ENGINEERED FEATURES (68 Base Concepts)            │
│ • Connectivity (15): Offline duration, disconnects     │
│ • Reboots (8): Reboot counts, reboot loops, durations  │
│ • System Health (6): CPU load, memory usage, uptime    │
│ • LoRa/Radio (8): RSSI, ECIO, packet drop rates        │
│ • Meter Reads (8): Read success %, trend, meters at risk│
│ • Gateway Info (8): Hardware age, antenna, firmware    │
│ • Visit History (5): Past fault rate, repeat offender  │
│ • Trends (10): 7d vs 28d degradation ratios            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ LightGBM CLASSIFIER & MULTI-MODEL BENCHMARK            │
│ • Asymmetric cost-weighted objective (FP=380, FN=600)  │
│ • Probability calibration (Platt scaling)              │
│ • Gateway-level GroupKFold CV (Zero leakage)           │
│ • AUC-ROC: 0.888 | Training time: ~10s                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ predictions.csv & EXPLAINABLE REASONS                  │
│ • Exactly 8 weeks × 15 visits/week (120 rows)          │
│ • Ranked by expected net financial savings             │
│ • SHAP feature attribution: Concise reasons (≤300 chars│
└────────────────────────────────────────────────────────┘
```

---

## 📋 All Available Commands (Windows & macOS / Linux)

Because Windows does not include `make` by default, use the dedicated table below for your operating system:

### 🪟 Windows (PowerShell / Command Prompt)
> **No `make` needed — all commands run via standard Python:**

| Task | Windows Command (PowerShell / CMD) | What It Does | Execution Time |
| :--- | :--- | :--- | :---: |
| **1. View Results & Cost** | `python scripts/evaluate.py` | Prints **0.888 AUC-ROC**, Accuracy, & **€30,380 Savings** vs 3σ Baseline | ~2s |
| **2. Open Dashboard** | `python scripts/dashboard.py` | Builds & automatically opens interactive dashboard in your browser | ~2s |
| **3. Full Pipeline** | `python scripts/predict.py --data data --out predictions.csv` | Auto-trains model (if needed) & generates 120-row `predictions.csv` | ~15s |
| **4. Validate Submission** | `python validate_submission.py predictions.csv` | Verifies `predictions.csv` against official grader rules (`OK`) | <1s |
| **5. Train Model Only** | `python scripts/train.py --data data --model-dir models` | Trains LightGBM model with cost-sensitive loss | ~10s |
| **6. Run All 55 Tests** | `pytest tests -q` | Runs complete unit & integration test suite (55 passed) | ~25s |
| **7. Multi-Model Benchmark**| `python scripts/compare_models.py` | Compares LightGBM, HistGradientBoosting, RandomForest & Logistic | ~15s |
| **8. Check Data Drift** | `python scripts/drift_check.py --data data --model-dir models` | Calculates Wasserstein distance & PSI drift metrics | ~5s |
| **9. Rollback Model** | `python scripts/rollback.py --to v1 --model-dir models` | Atomic rollback to previous model version | <1s |
| **10. Quick File Open** | `start dashboard.html` | Opens pre-built dashboard directly in Windows default browser | Instant |

---

### 🍎 🐧 macOS & Linux (Terminal / Make)
> **Run either via standard Python 3 or using the included Makefile shortcuts:**

| Task | macOS / Linux Command | Make Shortcut | Execution Time |
| :--- | :--- | :--- | :---: |
| **1. View Results & Cost** | `python3 scripts/evaluate.py` | `make evaluate` | ~2s |
| **2. Open Dashboard** | `python3 scripts/dashboard.py` | `make dashboard` | ~2s |
| **3. Full Pipeline** | `python3 scripts/predict.py --data data --out predictions.csv` | `make all` | ~15s |
| **4. Validate Submission** | `python3 validate_submission.py predictions.csv` | `make validate` | <1s |
| **5. Train Model Only** | `python3 scripts/train.py --data data --model-dir models` | `make train` | ~10s |
| **6. Run All 55 Tests** | `pytest tests/ -v` | `make test` | ~25s |
| **7. Multi-Model Benchmark**| `python3 scripts/compare_models.py` | — | ~15s |
| **8. Check Data Drift** | `python3 scripts/drift_check.py --data data --model-dir models` | `make drift` | ~5s |
| **9. Rollback Model** | `python3 scripts/rollback.py --to v1 --model-dir models` | `make rollback VERSION=v1` | <1s |
| **10. Clean Temp Files** | `rm -f predictions.csv baseline_predictions.csv` | `make clean` | <1s |

#### Docker Execution:
```bash
# Predict via Docker
docker compose up

# Run tests via Docker
docker compose --profile test up

# Train via Docker
docker compose --profile train up
```

---

## 📊 Interactive Dashboard

The system includes a self-contained executive HTML report (`dashboard.html`) designed for operations managers and engineering leads.

### How to Launch & View the Dashboard:
```bash
# Method 1: Generate & open automatically
make dashboard
# (or: python -c "from src.dashboard.report import main; main()")

# Method 2: Open directly in your browser
start dashboard.html        # Windows
open dashboard.html         # macOS

# Method 3: Run local HTTP server
python -m http.server 8765
# Navigate to: http://localhost:8765/dashboard.html
```

### Dashboard Key Features & Interactive Views:

| Section | What It Displays & How to Use It |
| :--- | :--- |
| **Fleet Health KPI Cards** | High-level fleet overview: **302 Safe vs 30 Broken** gateways, **€30,380** net savings, **0.888 AUC-ROC**, and **120** generated visit dispatches. |
| **1. Weekly Cost Comparison (Bar Chart)** | Side-by-side cost breakdown comparing the 3σ Baseline (€38,760/wk) against our Cost-Weighted Model (€30,920/wk) across all February evaluation weeks. |
| **2. Risk Score Distribution (Donut Chart)** | Categorizes all evaluated gateways into clear risk tiers: Critical Failure (90–100%), High Risk (70–89%), Moderate Risk (50–69%), and Stable Fleet (<50%). |
| **3. Failure Root Causes (Horizontal Bar Chart)** | Clear breakdown of detected faults across the fleet: Unstable Backhaul Drops (42%), Chronic Reboot Loops (25%), Radio Signal Degradation (18%), and Zero Meter Reads (15%). |
| **4. Cumulative Net Financial Savings (Trend Chart)** | Displays the accumulating operational savings over time, reaching **€30,380** saved across the 4-week window. |
| **Weekly Visit Schedule Table** | Filterable by week: displays the **Top 15 ranked gateways** with genuine risk score out of 100, specific root causes, and clear field actions (e.g. *Dispatch Technician for Antenna Replacement*). |
| **Next Week Early Warning Watchlist** | Highlights gateways ranked **#16 to #23** showing early degradation signs, allowing operations to pre-plan low-cost batch routes before complete failure. |

---

## 📈 How to Inspect Predictions & Model Benchmarks

### 1. View Predictions File:
The generated predictions are saved in standard CSV format:
```bash
# View top rows of predictions.csv
head -n 16 predictions.csv
```
Contains 5 columns: `week_start, rank, gateway_id, score, reason`.

### 2. Run Multi-Model Benchmark:
To see how our LightGBM model compares against other ML architectures:
```bash
python scripts/compare_models.py
```
Outputs cross-validated AUC-ROC, AUC-PR, Brier loss, and estimated costs across **LightGBM**, **HistGradientBoosting**, **RandomForest**, and **LogisticRegression**.

---

## 🔬 Feature Engineering (~130 Columns across 8 Domains)

| Group | Count | Key Features | Why It Matters |
| :--- | :---: | :--- | :--- |
| **Connectivity** | 15 | Offline duration, disconnection count, 3σ anomaly hours | Direct indicator of gateway availability and backhaul drops |
| **Reboots** | 8 | Reboot count, loop duration, power cycle frequency | Hardware instability and watchdog trip indicators |
| **System Health** | 6 | CPU load, memory exhaustion, process restarts | Resource leak and memory fragmentation signals |
| **LoRa/Radio** | 8 | RSSI, RSCP, ECIO, packet drop rates | Wireless transmission degradation and antenna faults |
| **Meter Reads** | 8 | Read success rate, 4-week trend, meters at risk | Primary business KPI and revenue impact indicator |
| **Gateway Info** | 8 | Installation age, firmware age, site type, meter count | Static baseline vulnerability factors |
| **Visit History** | 5 | Past visits, historical fault rate, days since last visit | Repeat offender detection and unresolved hardware bugs |
| **Trends** | 10 | 7-day ÷ 28-day ratio for all major metrics | Detects accelerating deterioration before total outage |

### Top 5 Most Predictive Features
| Rank | Feature | Why It Matters |
| :---: | :--- | :--- |
| 🥇 | `meters_at_risk` | Financial impact multiplier: failing gateways with 200+ meters cause severe revenue loss. |
| 🥈 | `read_rate_trend_4w` | Declining meter read percentage is the earliest leading indicator of hardware collapse. |
| 🥉 | `trend_offline_duration_sec_ratio` | Accelerating offline time signals failing cellular modem or antenna connection. |
| 4 | `7d_disconnection_cnt_mean` | Frequent short drops indicate network instability before complete disconnection. |
| 5 | `visit_fault_rate` | Repeat offenders: gateways with genuine past repair history have higher recurrent fault probability. |

---

## 🛡️ Data Integrity & Leakage Prevention

| Safeguard | How We Implement It |
| :--- | :--- |
| **Strict Temporal Cutoff** | Features for each Monday are computed **only using data prior to that Monday 00:00:00**. Zero lookahead leakage. |
| **Gateway-Level GroupKFold** | 5-fold cross-validation splits by `gateway_id`. The same physical gateway never appears in both train and validation sets. |
| **Ground Truth Isolation** | Engineer review (Feb 2026 ground truth) is strictly held out for evaluation and never used in training. |
| **Decommissioned Filtering** | Gateways decommissioned in a given week are filtered out based on active telemetry timestamps. |
| **Deterministic Seed** | Fixed random seed (`42`) used across all data splits, model seeds, and feature pipelines. |

---

## 📦 Model Versioning & Rollback

Every trained model is stored in an immutable, file-based model registry:

```text
models/
├── v1/
│   ├── model.joblib          # Trained LightGBM booster & calibrated classifier
│   ├── metadata.json         # Config, metrics, git commit SHA, timestamp
│   ├── feature_schema.json   # Exact feature columns and ordering
│   └── drift_reference.json  # Reference distribution statistics for drift checks
├── v2/
│   └── ...
└── current_version.txt       # Active pointer (e.g. "v1")
```

### Rollback to Previous Version:
```bash
# One-command rollback
make rollback VERSION=v1

# Verify output with rolled back version
make predict
make validate
```

---

## 🔍 Data Drift Detection

Run `make drift` to evaluate incoming data against baseline distributions:

| Check Type | Method | What It Catches |
| :--- | :--- | :--- |
| **Schema Drift** | Column presence, dtype validation | Missing or renamed telemetry fields |
| **Statistical Drift** | Wasserstein distance & PSI ($p < 0.05$) | Distribution shifts in disconnects, uptime, or signal |
| **Volume Drift** | Gateway count vs expected bounds | Mass gateway decommissioning or backhaul outages |

---

## 🐳 Docker Deployment

The system is containerized with a production multi-stage Docker build:

```bash
# Build the image
docker compose build

# Run prediction pipeline (default)
docker compose up

# Run full test suite
docker compose --profile test up

# Train model explicitly
docker compose --profile train up
```

- **Data is NOT baked into the image:** Mounted read-only (`./data:/app/data:ro`) satisfying challenge privacy rules.
- **Output Persisted:** `predictions.csv` and `models/` persist directly onto the host filesystem.

---

## ✅ Testing (55 Tests Passing)

Execute the complete test suite:
```bash
make test
```

| Test Suite File | Tests | What It Validates |
| :--- | :---: | :--- |
| `test_cost_function.py` | 7 | Asymmetric cost weights (€600 vs €380) & breakeven threshold ($P \approx 0.388$) |
| `test_data_drift.py` | 7 | Wasserstein distance, PSI drift scores, and warning tripwires |
| `test_data_loader.py` | 13 | Bare hex vs colon-separated ID normalization, Latin-1 fallback, missing data handling |
| `test_features.py` | 11 | 7d/28d rolling window aggregations, trend ratio calculation, NaN imputations |
| `test_generalization.py` | 6 | Stability on unseen gateway IDs and temporal robustness across Mondays |
| `test_model_registry.py` | 5 | Atomic version promotion, metadata tracking, and rollback consistency |
| `test_models.py` | 6 | Multi-model training and prediction (LightGBM, HistGradientBoosting, RandomForest, LogisticRegression) |

---

## 🗂️ Project Structure

```text
LPDG-RADIO-STATIONS/
├── README.md                      # Comprehensive project guide & documentation
├── DECISIONS.md                   # 5 key architecture decisions with trade-offs
├── AI-USAGE.md                    # AI disclosure and cost function correction
├── Dockerfile                     # Multi-stage production container build
├── docker-compose.yml             # Single-command execution profiles
├── Makefile                       # Developer automation targets
├── requirements.txt               # Pinned Python dependencies (100% reproducible)
├── config/
│   └── default.yaml               # Hyperparameters and business constants
├── predictions.csv                # Final validated predictions (120 rows)
├── baseline_3sigma.py             # Provided 3-sigma baseline script
├── validate_submission.py         # Official submission format checker
├── dashboard.html                 # Interactive executive dashboard
│
├── src/
│   ├── config.py                  # Environment variable configuration loader
│   ├── pipeline.py                # Train and predict pipeline orchestrator
│   ├── data/
│   │   ├── loader.py              # Parquet, CSV, and Excel data loaders
│   │   ├── validator.py           # Schema validation and type coercion
│   │   └── labels.py              # Multi-signal training label construction
│   ├── features/
│   │   ├── builder.py             # Feature pipeline coordinator
│   │   ├── telemetry_features.py  # 7d & 28d telemetry window aggregations
│   │   ├── meter_features.py      # Meter read success & trends
│   │   ├── gateway_features.py    # Hardware metadata & age
│   │   └── visit_features.py      # Historical technician visit outcomes
│   ├── model/
│   │   ├── cost_function.py       # Asymmetric business loss formulation
│   │   ├── trainer.py             # LightGBM & multi-model trainer
│   │   ├── predictor.py           # Inference engine
│   │   ├── ranker.py              # Cost-benefit gateway ranker
│   │   └── explainer.py           # SHAP-based natural language reason generator
│   ├── monitoring/
│   │   ├── data_drift.py          # Wasserstein and PSI drift monitors
│   │   ├── model_registry.py      # Version management & rollback engine
│   │   └── metrics_logger.py      # Performance telemetry logger
│   ├── dashboard/
│   │   └── report.py              # Plotly interactive dashboard generator
│   └── utils/
│       ├── gateway_ids.py         # Gateway ID normalizer (bare hex ↔ colon)
│       └── logging_setup.py       # Standardized structured logging
│
├── scripts/
│   ├── train.py                   # Model training entrypoint
│   ├── predict.py                 # Prediction generation entrypoint
│   ├── evaluate.py                # Cost evaluation vs baseline
│   ├── compare_models.py          # Multi-model benchmarking script
│   ├── drift_check.py             # Data drift CLI runner
│   └── rollback.py                # Version rollback CLI runner
│
├── tests/                         # 55 unit & integration tests
└── models/                        # Versioned model artifacts (gitignored)
```

---

## ⚙️ Configuration

All configuration values can be adjusted in `config/default.yaml` or overridden via environment variables:

| Setting | Config Key | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :---: | :--- |
| **Data Directory** | `paths.data_dir` | `DATA_DIR` | `./data` | Directory containing challenge dataset |
| **Model Directory**| `paths.model_dir` | `MODEL_DIR` | `./models` | Directory storing versioned models |
| **Predictions Out**| `paths.predictions_out` | `PREDICTIONS_OUT` | `./predictions.csv` | Output path for grader |
| **Visit Cost** | `cost.visit_cost` | `COST_FALSE_POSITIVE` | `380` | Cost of technician visit (€) |
| **Miss Cost** | `cost.miss_cost` | `COST_FALSE_NEGATIVE` | `600` | Weekly penalty for broken gateway (€) |
| **Visit Budget** | `cost.visit_budget` | `VISIT_BUDGET` | `15` | Max technician visits per week |
| **Random Seed** | `model.random_seed` | `RANDOM_SEED` | `42` | Deterministic random seed |
| **Log Level** | `logging.level` | `LOG_LEVEL` | `INFO` | Console logging verbosity |

---

## 🎓 Competency Areas Covered

| Competency Area | What We Demonstrate in This Implementation |
| :--- | :--- |
| **E — Machine Learning (Primary)** | Cost-sensitive LightGBM classifier, ~130 engineered features, GroupKFold CV, SHAP explainability, multi-model benchmark, beats baseline by €30,380. |
| **D — Data Science** | Rigorous economic evaluation framework, multi-signal ground truth synthesis, temporal trend analysis, and interactive executive reporting dashboard. |
| **F — MLOps** | File-based model registry, atomic rollback CLI, automated schema/distribution drift detection, and reproducible training pipeline. |
| **C — DevOps (Docker)** | Multi-stage containerization, read-only data mounting, 12-factor environment configuration, and GitHub Actions CI workflow. |

---

## 📝 Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for full rationale and alternatives considered.

| Decision | Summary Rationale |
| :--- | :--- |
| **LightGBM over Deep Learning** | Fast training (~10s, critical for live evaluation), handles missing data natively, superior on tabular telemetry. |
| **Cost-Weighted Loss ($FP=380, FN=600$)** | Optimizes total monetary cost rather than generic accuracy, aligning directly with utility business incentives. |
| **GroupKFold CV over Random Split** | Prevents data leakage by ensuring the same gateway never appears in both training and validation sets. |
| **Domain Feature Engineering over Raw Series** | 7d/28d trend ratios filter high-frequency cellular noise and isolate genuine physical hardware degradation. |
| **Self-Contained File Registry over MLflow** | Lightweight, zero external database dependencies, works offline and inside simple Docker containers. |

---

## 📄 Notice & License

This project was built for the **LPDG Innovation Hub Selection Challenge 2026**. All challenge telemetry and operational data remain the confidential property of the challenge organizers.

**Author & Maintainer:** [keerthi machanooru](https://github.com/Keerthimk24) ([@Keerthimk24](https://github.com/Keerthimk24))
