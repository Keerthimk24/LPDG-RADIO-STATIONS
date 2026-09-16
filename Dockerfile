# ============================================================================
# Stage 1: Builder — install dependencies in an isolated layer
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================================
# Stage 2: Runtime — lean production image
# ============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Data and models are MOUNTED, never baked into the image
VOLUME /app/data
VOLUME /app/models

# All configuration via environment variables
ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/models
ENV PREDICTIONS_OUT=/app/predictions.csv
ENV LOG_LEVEL=INFO
ENV RANDOM_SEED=42
ENV VISIT_BUDGET=15
ENV COST_FALSE_POSITIVE=380
ENV COST_FALSE_NEGATIVE=600
ENV MODEL_VERSION=current

# Health check: validates config and data availability
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.config import Config; Config.validate()" || exit 1

# Default: run predictions
ENTRYPOINT ["python"]
CMD ["scripts/predict.py"]
