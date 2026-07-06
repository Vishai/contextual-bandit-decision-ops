.PHONY: install test lint check smoke demo release-check docker-build docker-smoke

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

release-check: check smoke
	contextual-bandit-info --version
	bash -n scripts/generate_demo.sh
	bash -n scripts/docker_smoke_test.sh
	test -f CHANGELOG.md
	test -f LICENSE
	test -f CITATION.cff
	test -f docs/architecture.md
	test -f docs/policy_card.md
	test -f docs/evaluation_methodology.md
	test -f docs/operations.md
	test -f docs/release_checklist.md
	test -f docs/interview_notes.md
	test -f reports/portfolio/release_0.1.0.md

docker-build:
	docker build --tag "$(IMAGE_NAME)" .

docker-smoke:
	IMAGE_NAME="$(IMAGE_NAME)" bash scripts/docker_smoke_test.sh
