install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check: test lint
