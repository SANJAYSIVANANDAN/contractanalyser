# Single image for the whole application: the FastAPI api, the Celery worker, and the
# Streamlit frontend are all the same image, started with different `command:` overrides
# in docker-compose.yml. No Dockerfile.worker/.frontend/.eval, no Alembic layer.
FROM python:3.12-slim
WORKDIR /app

# CPU-only PyTorch + torchvision together, from the CPU wheel index (the local reranker +
# Docling need them; the default CUDA build is huge). Installing them in the SAME command is
# what matters: torch and torchvision are ABI-matched release pairs, and letting pip resolve
# torchvision separately later (e.g. as a dependency of sentence-transformers/docling) can pull
# in a mismatched build that crashes at import time with "operator torchvision::nms does not
# exist".
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
