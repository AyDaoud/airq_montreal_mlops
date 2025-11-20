.PHONY: install lint format test run-api run-flow build-docker run-docker

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

build-docker:
	docker build -t airq-api .

run-docker:
	docker run --rm -p 8000:8000 \
		-e MODEL_PATH=/app/artifacts/rf/model.pkl \
		airq-api
