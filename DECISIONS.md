# Decisions

Five choices I made, and for each one, what else I could have done and why I did not.

---

## 1. Area Choice: Machine Learning (primary), supported by Data Science & MLOps

**What I chose:** Machine Learning as my primary area, with Data Science providing the analytical foundation and MLOps ensuring the model is production-ready.

**What else I could have done:**
- **Pure Data Science:** Focus entirely on the statistical analysis and cost optimisation without building an ML model. The baseline 3-sigma approach works, and I could have spent all my time on the threshold analysis and operations report.
- **Pure MLOps:** Take the baseline model as-is and focus entirely on containerisation, CI/CD, monitoring, and versioning.
- **Software Development:** Wrap the baseline in a web API with tests and clean architecture.

**Why I chose this:** The 61% false-positive rate in the current field visit data (390 out of 642 visits found nothing wrong) told me the biggest impact comes from a smarter ranking model. The baseline's 3-sigma rule treats every gateway the same — it does not know that a gateway with 300 meters matters more than one with 40, or that a gateway with a history of real faults is more likely to need attention again. An ML model can learn these patterns. I supported it with Data Science (to define "needs a visit" and analyse costs honestly) and MLOps (to make the model reproducible and rollback-safe) because a model you cannot trust or reproduce is not useful.

---

## 2. Defining "Needs a Visit" as Cost-Weighted Degradation, Not Binary Anomaly

**What I chose:** A gateway "needs a visit" if it shows reliability degradation (rising disconnections, offline time, reboots) that is persistent (>7 days), impacts meter reads, and where the cost of leaving it exceeds the cost of sending someone.

**What else I considered:**
- **Any anomaly above 3 sigma:** This is the baseline approach. It flags any statistical outlier without considering impact or persistence. It catches noise — a single bad hour can flag a gateway.
- **Meter read rate below a threshold (e.g., <50%):** This is high-precision but low-recall — by the time meter reads drop below 50%, the gateway has been failing for weeks and the €600/week cost has already accumulated.
- **Engineer says "Schlecht":** Using the binary classification directly. This ignores cost asymmetry — we need a ranked list of 15, not a yes/no answer.

**Why I chose this:** The cost structure (€380 per visit, €600 per week if broken) means we need to catch problems early, but not so aggressively that we waste visits on noise. My definition combines leading indicators (telemetry trends) with lagging confirmation (meter reads) and weights by impact (number of meters behind the gateway).

---

## 3. LightGBM with Custom Asymmetric Cost Function Instead of Standard Classification

**What I chose:** LightGBM with a custom objective function that penalises false negatives (missing a broken gateway, €600/week) more heavily than false positives (unnecessary visit, €380).

**What else I considered:**
- **Standard binary cross-entropy:** Treats false positives and false negatives equally. This would optimise for accuracy, not for cost. A model that predicts "Normal" for everything would be 80%+ accurate but useless.
- **Simple `scale_pos_weight`:** LightGBM's built-in class weighting. Simpler to implement, but less precise — it does not directly encode the euro amounts.
- **XGBoost or Random Forest:** Similar capabilities. XGBoost would work equally well. Random Forest does not support custom objectives as cleanly.
- **Neural network:** Overkill for 332 gateways and tabular data. Would be harder to explain and slower to retrain in the live session.

**Why I chose this:** The brief says "A model that beats baseline_3sigma.py on total cost. Not on accuracy." A custom cost function directly optimises what we are measured on. LightGBM is fast (trains in seconds, critical for the live session), handles mixed feature types natively, and supports SHAP for explainability.

---

## 4. Multi-Signal Label Construction Instead of Single Ground Truth

**What I chose:** Constructed training labels from three signals:
1. **Engineer review** (strongest): 60 "Schlecht" + 60 "Normal" gateways, reviewed 15 Feb 2026
2. **Field visit outcomes** (medium): gateways where a visit found and fixed a fault → genuinely needed attention
3. **Meter read collapse** (proxy): read success rate below 50% → likely hardware issue

**What else I considered:**
- **Engineer review only:** Only 120 labelled gateways out of 320. Training on just these would mean small sample size and potential bias (the engineer only reviewed a subset).
- **Unsupervised anomaly detection:** No labels at all — use isolation forest or autoencoders. This avoids the labelling problem but cannot incorporate the cost structure and is harder to explain.
- **Field visit outcomes only:** 642 visits, but 61% found nothing wrong. Using "was visited" as a positive label would teach the model to replicate the current (bad) decision-making.

**Why I chose this:** Each signal has weaknesses. The engineer review is small but expert. Visit outcomes are noisy but real. Meter reads are delayed but objective. Combining them gives more coverage (more labelled gateways) and more robustness (no single signal can bias the model too much). For gateways with no signal, I default to "Normal" because the base rate of problems is low.

---

## 5. Separate Training and Prediction Pipelines with Versioned Model Registry

**What I chose:** Training (`scripts/train.py`) and prediction (`scripts/predict.py`) are completely separate scripts. Each model version is saved with its weights, metadata (training parameters, feature list, git SHA, timestamp), and a feature schema. Rollback is a one-command operation (`scripts/rollback.py --to v1`).

**What else I considered:**
- **Single script that trains and predicts:** Simpler, fewer files. But it means you cannot predict without retraining, which is slow and risky in the live session. The brief explicitly says "keep training separate from predicting — anything that retrains while it is trying to answer will not finish in the room."
- **MLflow or Weights & Biases:** Full experiment tracking platforms. Powerful but heavy dependencies that would complicate the Docker setup and are overkill for a single-model project.
- **Database-backed registry:** PostgreSQL or SQLite for version management. More robust for a team, but unnecessary complexity for a single developer.

**Why I chose this:** The brief tests whether you can change things with people watching. Separate pipelines mean I can retrain with new data in one terminal while the old model keeps serving predictions. The file-based registry is simple, has no external dependencies, and the rollback is tested (there is a test in `test_model_registry.py` that trains v1, trains v2, rolls back to v1, and verifies predictions match).

---

## What It Cannot Do

1. **Cannot predict sudden catastrophic hardware failures** — The model identifies gradual degradation trends (rising disconnections, memory leakage, reboot clusters, declining read success). A physical surge, direct lightning strike, or power drop off is not detectable in advance from telemetry.
2. **Cannot account for external environmental events** — Cellular carrier base station outages, extreme weather, or regional grid maintenance are absent from the training data.
3. **Subject to label noise** — Labels combine engineer review (120 samples) and field visit notes (61% false positive baseline rate). Ground truth on the exact universe of failing gateways remains partially unobserved.
4. **Calibrated for a fixed budget constraint** — Thresholds and rankings are tailored to the strict 15 visits/week operational capacity. Changing capacity to 30 or 5 shifts the optimal operating threshold.
5. **Batch weekly inference, not streaming real-time alerts** — Evaluates state on Mondays for the upcoming week; does not generate real-time intraday push alerts.

---

## What Another Two Weeks Would Fix

1. **Hyperparameter Optimisation with Optuna** — Automated Bayesian search across tree depth, learning rate, feature fractions, and regularization penalties instead of manual grid tuning.
2. **Hybrid Ensemble with 3-Sigma Rules** — Blend LightGBM continuous risk probabilities with 3-sigma anomaly tripwires to catch acute spikes that trees occasionally smooth out.
3. **Automated Live A/B Testing & Shadow Evaluation** — Continuous scoring in shadow mode against live field dispatches with automatic drift-triggered retraining gates.
4. **Real-Time Streaming Anomaly Engine** — Event-driven Kafka / Faust pipeline computing running z-scores and alerting field engineers within 1 hour of silent gateway collapse.
