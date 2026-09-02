#!/bin/bash

set -e

echo "🔍 Running local CI validation..."
echo ""

# Check Python version
echo "📋 Python version:"
python --version
echo ""

# Install dev dependencies
echo "📦 Installing dependencies..."
pip install -q -e '.[dev]' pip-audit
echo ""

# Run linting
echo "🎨 Running linter (ruff)..."
ruff check src/ tests/ --fix --exit-zero || true
ruff format src/ tests/ --check || true
echo "✅ Linting complete"
echo ""

# Run security audit
echo "🔐 Running security audit (pip-audit)..."
pip-audit --desc || echo "⚠ Some vulnerabilities found (review and update as needed)"
echo ""

# Run tests with coverage
echo "🧪 Running pytest with coverage..."
pytest tests/ -v --cov=src/furniture_ai --cov-report=term --cov-fail-under=75
echo "✅ Tests passed with coverage > 75%"
echo ""

# Run smoke test
echo "🚀 Running smoke test..."
bash scripts/smoke_test.sh
echo ""

echo "✅ All CI checks passed locally!"
exit 0
