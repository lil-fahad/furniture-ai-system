.PHONY: install test lint run ui docker check audit models-check models-verify
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

docker:
	docker compose up --build

models-check:
	python scripts/install_professional_bundle.py --check-spec
	python scripts/model_manifest.py

models-verify:
	python scripts/install_professional_bundle.py --verify-installed

audit:
	python scripts/scan_secrets.py
	python scripts/repository_audit.py

check: lint test models-check audit
