.PHONY: run train test predict validate evaluate clean help

help: ## Show this help
	@echo LPDG Gateway Health Predictor
	@echo =============================
	@echo.
	@echo Available targets:
	@echo   make run       - Generate predictions (docker)
	@echo   make train     - Train the model (docker)
	@echo   make test      - Run all tests (docker)
	@echo   make predict   - Generate predictions + validate (local)
	@echo   make validate  - Validate predictions.csv
	@echo   make evaluate  - Evaluate predictions against ground truth
	@echo   make clean     - Remove Docker images and artifacts

run: ## Generate predictions via Docker (default)
	docker compose up --build gateway-predictor

train: ## Train the model via Docker
	docker compose --profile train up --build train

test: ## Run all tests via Docker
	docker compose --profile test up --build test

predict: ## Generate predictions locally + validate
	python scripts/train.py --data ./data --model-dir ./models
	python scripts/predict.py --data ./data --out predictions.csv
	python validate_submission.py predictions.csv

validate: ## Validate predictions.csv format
	python validate_submission.py predictions.csv

evaluate: ## Evaluate predictions against ground truth
	python scripts/evaluate.py --predictions predictions.csv --data ./data

clean: ## Remove Docker images and generated artifacts
	docker compose down --rmi local
	@if exist models rmdir /s /q models
	@if exist predictions.csv del predictions.csv
