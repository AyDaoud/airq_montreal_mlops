.PHONY: install lint format test run-api run-flow bake-model build-docker build-docker-train run-docker setup ingest build-data data

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

lint:
	flake8 src tests

format:
	black src tests

run-api:
	uvicorn src.serving.app:app --reload

run-flow:
	python -m orchestration.flow

bake-model:
	$(PY) -m scripts.bake_serving_model

build-docker: bake-model
	docker build -t airq-api .

build-docker-train:
	docker build -f Dockerfile.train -t airq-train .

run-docker:
	docker run --rm -p 8000:8000 \
		-e MODEL_PATH=/app/artifacts/rf/model.pkl \
		airq-api

PY := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m src.data.cli ingest

build-data:
	$(PY) -m src.data.cli build

data: ingest build-data
