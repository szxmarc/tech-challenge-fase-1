.PHONY: lint test run

lint:
	python -m ruff check src/

test:
	python -m pytest tests/ -v

run:
	python run_app.py