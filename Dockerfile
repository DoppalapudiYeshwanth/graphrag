# Graph RAG API image. CPU-only PyTorch keeps it small; the embedding model is baked in
# so the container starts without downloading anything.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

COPY requirements-api.txt .
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

# Bake the embedding model into the image (must match EMBEDDING_MODEL).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY graphrag ./graphrag

RUN useradd --create-home app && chown -R app /app /opt/hf
USER app

# Cloud Run injects PORT; default for local runs.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn graphrag.api:app --host 0.0.0.0 --port ${PORT}"]
