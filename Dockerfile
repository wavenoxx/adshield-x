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
COPY outputs/adshield_model.joblib ./outputs/adshield_model.joblib

# The database lives on the container filesystem, which most free hosts wipe on
# redeploy. Point ADSHIELD_DB at a mounted volume if history must survive.
ENV ADSHIELD_DB=/app/outputs/adshield.db

EXPOSE 7860
CMD gunicorn --chdir app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 \
    --timeout 120 "app:app"
