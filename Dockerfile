# Serving image: sklearn inference only - no torch, no prophet, no mlflow.
# Requires artifacts/rf to exist: run `python -m scripts.bake_serving_model` first.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-serving.txt

COPY src/__init__.py ./src/
COPY src/serving ./src/serving
COPY src/features ./src/features
COPY artifacts/rf ./artifacts/rf

ENV MODEL_PATH=/app/artifacts/rf/model.pkl
ENV FEATURES_PATH=/app/artifacts/rf/feature_names.json
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
