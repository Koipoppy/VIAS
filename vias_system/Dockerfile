FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIAS_DB_PATH=/data/vias.db \
    VIAS_STORAGE_DIR=/data/uploads \
    VIAS_TEXT_MODEL_DIR=/app/models

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models
COPY docs ./docs

RUN mkdir -p /data/uploads
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "vias.main:app", "--host", "0.0.0.0", "--port", "8000"]
