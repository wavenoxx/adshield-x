# AdShield-X console. Builds the Flask app with the trained model baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# libgomp is needed by xgboost and lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY docs/ ./docs/
# The console needs two artefacts at runtime: the trained bundle, and the
# held-out stream that "generate a sample" draws from. Everything else under
# outputs/ is paper material and stays out of the image.
COPY outputs/adshield_model.joblib ./outputs/adshield_model.joblib
COPY outputs/sample_pool.csv       ./outputs/sample_pool.csv

# The database lives on the container filesystem, which most free hosts wipe on
# redeploy. Point ADSHIELD_DB at a mounted volume if history must survive.
ENV ADSHIELD_DB=/app/outputs/adshield.db

EXPOSE 7860
CMD gunicorn --chdir app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 \
    --timeout 120 "app:app"
