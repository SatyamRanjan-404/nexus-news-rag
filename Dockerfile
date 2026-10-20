# Multi-stage production Dockerfile for Nexus News RAG
# Supports both 'api' and 'ui' targets

FROM python:3.10-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling 500MB CUDA build)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . .
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/chroma_db && \
    chown -R appuser:appuser /app
USER appuser

# ── API Target (FastAPI) ──────────────────────────────────────────
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── UI Target (Streamlit) ─────────────────────────────────────────
FROM base AS ui
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/nexus/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]