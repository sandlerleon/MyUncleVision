# MyUncleVision: one command, no manual step.
#   docker build -t myuncle-vision .
#   docker run --rm -v /path/to/svs:/input:ro -v /path/to/results:/output myuncle-vision
# Input: a directory of whole-slide images (SVS).  Output: findings, per-slide annotations / masks / heat maps / patch scores, images_used.csv, run_manifest.json.
# GPU: build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 and run with --gpus all.
FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch torchvision --index-url ${TORCH_INDEX} && pip install -r requirements.txt
COPY . .
# bake the DINOv2 weights (timm vit_small_patch14_dinov2.lvd142m, about 88 MB) into the image so that the run needs no network
RUN python -m myuncle_vision.encoder --prefetch
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python", "-m", "myuncle_vision"]
CMD ["run", "--input", "/input", "--output", "/output"]
