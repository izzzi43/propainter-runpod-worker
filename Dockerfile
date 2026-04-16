FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ProPainter
RUN git clone --depth 1 https://github.com/sczhou/ProPainter.git /app/ProPainter

# Download model weights (~700MB total)
RUN mkdir -p /app/ProPainter/weights && \
    cd /app/ProPainter/weights && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH="/app/ProPainter:/app/src:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/src/handler.py"]
