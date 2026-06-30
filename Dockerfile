# ─────────────────────────────────────────────────────────────────────────────
# Exoplanet Classifier — Streamlit dashboard container
# Build:  docker build -t exoplanet-classifier .
# Run:    docker run -p 8501:8501 exoplanet-classifier
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries needed by lightgbm / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The trained model is committed, so just serve the dashboard. (To retrain inside
# the image instead, install requirements-dev.txt and run `python -m src.train`.)
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
