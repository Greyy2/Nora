# Dockerfile for Grey Backend (FastAPI + Vectorized Numpy)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY backend/requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir \
            fastapi \
            uvicorn[standard] \
            pandas \
            numpy \
            pymongo \
            scipy \
            scikit-learn \
            python-multipart \
            openpyxl \
            tqdm \
            fire \
            loguru \
            python-dotenv \
            pydantic-settings \
            rdagent==0.8.0; \
    fi

# Code will be MOUNTED from host (not COPIED)
# This allows hot reload without rebuild

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV GREY_FAST_MODE=true

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
