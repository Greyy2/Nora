# SOVA Qlib Verification - Minimal CPU-Only Image
# Resource-limited: no GPU, minimal packages, just enough to run Qlib factor eval
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Qlib from pinned commit (same as RD-Agent)
RUN git clone https://github.com/microsoft/qlib.git /workspace/qlib && \
    cd /workspace/qlib && \
    git fetch && git reset 2fb9380b342556ddb50a4b24e4fe8655d548b2b8 --hard && \
    pip install --no-cache-dir cython && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir lightgbm tables pyarrow

WORKDIR /workspace/eval

COPY eval_factors.py /workspace/eval/
COPY conf_sova_factors.yaml /workspace/eval/
COPY read_results.py /workspace/eval/

CMD ["python", "eval_factors.py"]
