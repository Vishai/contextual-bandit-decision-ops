.PHONY: install test lint check smoke demo docker-build docker-smoke

PYTHON ?= python
IMAGE_NAME ?= contextual-bandit-decision-ops:local

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check: test lint

smoke:
	contextual-bandit-service --smoke-test

demo:
	bash scripts/generate_demo.sh

docker-build:
	docker build --tag "$(IMAGE_NAME)" .

docker-smoke:
	IMAGE_NAME="$(IMAGE_NAME)" bash scripts/docker_smoke_test.sh
