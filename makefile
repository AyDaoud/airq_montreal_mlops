.PHONY: install lint format test run-api run-flow bake-model build-docker build-docker-train run-docker setup ingest build-data data

# All recipes run through the venv interpreter. Bare `pytest`/`flake8`/
# `uvicorn` are not on PATH after `make setup`, which silently broke the
# README quickstart on a fresh clone.
PY := .venv/bin/python

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m flake8 src tests scripts

format:
	$(PY) -m black src tests scripts

run-api:
	$(PY) -m uvicorn src.serving.app:app --reload

run-flow:
	$(PY) -m orchestration.flow

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

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m src.data.cli ingest

build-data:
	$(PY) -m src.data.cli build

data: ingest build-data
