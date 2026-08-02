.PHONY: install test lint run ui check
install:
	python -m pip install -e ".[dev,ui]"

test:
	pytest

lint:
	ruff check .

run:
	uvicorn furniture_ai.api:app --reload

ui:
	streamlit run apps/streamlit_app.py

check: lint test
