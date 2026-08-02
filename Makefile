.PHONY: setup sync-public sync-all test lint api

setup:
	python -m pip install -e ".[dev]"

sync-public:
	bash scripts/sync_sources.sh --public-only

sync-all:
	bash scripts/sync_sources.sh --all

test:
	pytest

lint:
	ruff check .

api:
	uvicorn furniture_system.main:app --host 0.0.0.0 --port 8000 --reload
