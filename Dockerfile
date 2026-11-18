FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SRT_DEMO=1 \
    SRT_ALLOW_FORTRAN=0 \
    SRT_MAX_NZ=20 \
    SRT_MAX_ITER=300 \
    SRT_MAX_WORKERS=1 \
    SRT_RATE_LIMIT=40 \
    SRT_RATE_WINDOW=600 \
    SRT_RUNS_DIR=/tmp/srt_runs \
    SRT_JAX_CACHE=/tmp/srt_jax_cache \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY diff_srt ./diff_srt
COPY launcher.py README.md ./

EXPOSE 7860

# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
CMD ["python", "launcher.py", "--headless", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
