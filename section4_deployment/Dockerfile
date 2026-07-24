# Dockerfile for the LLM inference API
# 
# Build:  docker build -t llm-api .
# Run:    docker run --gpus all -p 8000:8000 llm-api
#
# Note: model is downloaded at first startup (not baked into image)
# This keeps the image smaller (~4GB vs ~7GB) at the cost of cold start time.
# In prod I'd use a pre-warmed image or mount model weights from a volume.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    USE_4BIT=true \
    MAX_NEW_TOKENS=256 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python3", "app.py"]
