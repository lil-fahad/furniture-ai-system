.PHONY: install test lint run web web-build ui-legacy docker check audit models-check models-verify
install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

run:
	uvicorn furniture_ai.api_entry:app --reload

web:
	cd apps/web && npm install && npm run dev

web-build:
	cd apps/web && npm install --ignore-scripts --no-fund --no-audit && npm run typecheck && npm run build

ui-legacy:
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
