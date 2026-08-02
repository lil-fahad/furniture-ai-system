FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models/manifest.json ./models/manifest.json
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "furniture_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
