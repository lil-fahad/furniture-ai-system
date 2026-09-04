FROM python:3.12-slim
LABEL org.opencontainers.image.title="furniture-ai-system" \
      org.opencontainers.image.version="1.4.0"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The wheel lands in site-packages, so PROJECT_ROOT-based anchoring cannot
# find the shipped data/models; pin absolute runtime paths instead. /data is
# created below and chowned to the app user.
ENV MODEL_MANIFEST_PATH=/app/models/manifest.json \
    DATABASE_PATH=/data/furniture_ai.sqlite3 \
    CATALOG_PATH=/app/data/furniture_catalog.json
WORKDIR /app
# opencv-python-headless needs no system GL libraries; keep the image minimal.
RUN groupadd --system app && useradd --system --gid app --no-create-home app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models/manifest.json ./models/manifest.json
# The package is installed from the vendored source tree, so the build is
# byte-exact and reproducible (equivalent to pinning furniture-ai-system==1.4.0).
RUN pip install --no-cache-dir . \
    && mkdir -p /data \
    && chown -R app:app /app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "furniture_ai.api_entry:app", "--host", "0.0.0.0", "--port", "8000"]
