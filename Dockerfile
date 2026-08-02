FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY sources.lock.json ./

RUN python -m pip install --upgrade pip && python -m pip install .

USER 65532:65532
EXPOSE 8000

CMD ["uvicorn", "furniture_system.main:app", "--host", "0.0.0.0", "--port", "8000"]
