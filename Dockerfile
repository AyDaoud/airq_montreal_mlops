# Dockerfile
FROM python:3.12-slim

# System deps for Prophet / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy code + artifacts
COPY src ./src
COPY artifacts ./artifacts

# Defaults – can be overridden at `docker run`
ENV MODEL_PATH=/app/artifacts/rf/model.pkl
ENV FEATURE_COLUMNS=""

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
