.PHONY: install test lint run ui docker check audit models-check models-verify style-prepare style-train
install:
	python -m pip install -e ".[dev,ui]"

test:
	pytest

lint:
	ruff check .

run:
	uvicorn furniture_ai.api_entry:app --reload

ui:
	streamlit run apps/streamlit_app.py

docker:
	docker compose up --build

models-check:
	python scripts/install_professional_bundle.py --check-spec
	python scripts/model_manifest.py

models-verify:
	python scripts/install_professional_bundle.py --verify-installed

style-prepare:
	python scripts/prepare_style_dataset.py data/styles \
		--source-manifest data/style_sources.jsonl \
		--output data/styles_prepared

style-train:
	python training/train_style_classifier.py data/styles_prepared \
		--output models/style_classifier/efficientnet_b0.pth

audit:
	python scripts/scan_secrets.py
	python scripts/repository_audit.py

check: lint test models-check audit
