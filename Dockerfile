FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATA_DIR=/data FASTEMBED_CACHE_PATH=/models

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

RUN python -c "from app.embeddings import FastEmbedEmbedder; from app.config import Settings; FastEmbedEmbedder(Settings().embedding_model)"

COPY static ./static

VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
