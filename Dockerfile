# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies for building certain python packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime system dependencies for soundfile and openvino
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy source code and other necessary files
COPY src/ ./src/
COPY README.md .

# Ensure permissions for non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Healthcheck to verify the container is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import torch; import openvino; print('Ready')" || exit 1

# Default command
CMD ["python3", "src/pipeline.py"]