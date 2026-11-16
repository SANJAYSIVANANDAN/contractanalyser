FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch first (the local reranker needs it; the default CUDA build is huge)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
