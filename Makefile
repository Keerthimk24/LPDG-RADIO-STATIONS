.PHONY: all run train test predict validate evaluate dashboard drift rollback clean help

help: ## Show this help
	@echo LPDG Gateway Health Predictor
	@echo =============================
	@echo.
	@echo Available targets:
	@echo   make all       - Train + Predict + Validate (default)
	@echo   make run       - Generate predictions (docker)
	@echo   make train     - Train the model
	@echo   make predict   - Generate predictions + validate
	@echo   make validate  - Validate predictions.csv
	@echo   make evaluate  - Evaluate predictions against ground truth
	@echo   make dashboard - Generate interactive HTML dashboard
	@echo   make test      - Run all tests
	@echo   make drift     - Run data drift detection
	@echo   make rollback  - Rollback model (usage: make rollback VERSION=v1)
	@echo   make clean     - Remove Docker images and artifacts

all: train predict validate ## Train, predict, and validate (default)

run: ## Generate predictions via Docker (default)
	docker compose up --build gateway-predictor

train: ## Train the model
	python scripts/train.py --data ./data --model-dir ./models

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

predict: ## Generate predictions + validate
	python scripts/predict.py --data ./data --out predictions.csv
	python validate_submission.py predictions.csv

validate: ## Validate predictions.csv format
	python validate_submission.py predictions.csv

evaluate: ## Evaluate predictions against ground truth
	python scripts/evaluate.py --predictions predictions.csv --data ./data

dashboard: ## Generate interactive HTML dashboard
	python -c "from src.dashboard.report import main; main()"

drift: ## Run data drift detection
	python scripts/drift_check.py --data ./data --model-dir ./models

rollback: ## Rollback to a previous model version (usage: make rollback VERSION=v1)
	python scripts/rollback.py --to $(VERSION) --model-dir ./models

clean: ## Remove generated artifacts
	@if exist models rmdir /s /q models
	@if exist predictions.csv del predictions.csv
	@if exist dashboard.html del dashboard.html
	@if exist baseline_predictions.csv del baseline_predictions.csv
