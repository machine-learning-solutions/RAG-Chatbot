# CPU PyTorch — Ollama owns the GPU on 8 GB cards. Image stays CUDA-capable for larger GPUs.
FROM python:3.11-slim

RUN pip install --no-cache-dir torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY app ./app

RUN mkdir -p /app/data/uploads /app/data/vector_store

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
