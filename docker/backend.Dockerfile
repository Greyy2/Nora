# Grey Backend Dockerfile - Production Optimized
FROM python:3.12-slim

# Metadata
LABEL maintainer="Grey Optimization Team"
LABEL version="1.0"
LABEL description="Grey Vectorized Backtest Engine"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY backend/requirements.txt .

# Install core backend packages first so the API can boot even when optional
# AI/Quanta dependencies in the full requirements set conflict.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.109.0 \
        "uvicorn[standard]==0.27.0" \
        pymongo==4.6.1 \
        motor==3.3.2 \
        "redis>=5.0.0" \
        "pandas>=2.3.0" \
        "numpy>=2.0.0" \
        "polars>=1.0.0" \
        "pyarrow>=18.0.0" \
        python-multipart==0.0.6 \
        pydantic==2.5.3 \
        pydantic-settings==2.1.0 \
        openpyxl==3.1.2 \
        xlsxwriter==3.2.9 \
        tqdm==4.67.1 \
        psutil==5.9.8 \
        gspread==6.0.0 \
        google-auth==2.27.0 \
        google-auth-oauthlib==1.2.0 \
        google-api-python-client==2.160.0 \
        fire \
        loguru \
        filelock \
        fuzzywuzzy \
        openai \
        python-dotenv \
        pandarallel \
        scikit-learn \
        tiktoken \
        pymupdf \
        dill \
        tables \
        docker \
        setuptools-scm \
        matplotlib \
        plotly \
        scipy \
        lightgbm \
        pyyaml \
        typing-extensions

# Copy application code
COPY backend/ ./backend/

# Create cache directory
RUN mkdir -p /app/cache /app/data

# Create non-root user for security
RUN useradd -m -u 1000 greyuser && \
    chown -R greyuser:greyuser /app

# Switch to non-root user
USER greyuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Environment variables (can be overridden in docker-compose)
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV WORKERS=39

# Start application
# Start application with PYTHONPATH set to /app/backend to resolve internal imports
ENV PYTHONPATH=/app/backend
CMD ["sh", "-c", "python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000 --workers 1"]

# kill: pkill -f python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000 --workers 1
