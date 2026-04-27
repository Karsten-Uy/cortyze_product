# GPU worker image for RunPod (Pod or Serverless).
#
# Build:   make image-build  (or docker buildx build --platform linux/amd64 ...)
# Push:    same target with --push
# Size:    ~3-5 GB (model weights are NOT baked in — they live on a network
#          volume mounted at /opt/hf_cache, which is faster to populate once
#          and cheaper to share across pods than re-pulling on every cold start).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1

WORKDIR /app

# System tools TRIBE v2 shells out to: ffmpeg for audio extraction, uv/uvx
# for the whisperx transcription step (tribev2/eventstransforms.py invokes
# `uvx whisperx ...` instead of importing whisperx directly to keep its deps
# isolated from the runtime env).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

# Install deps in a separate layer so code changes don't bust the cache.
COPY gpu_worker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Worker code + the cortyze packages it imports from
COPY gpu_worker/ /app/gpu_worker/
COPY services/ /app/services/
COPY core/ /app/core/

# PYTHONPATH down here so it lives below the slow pip-install layer; tweaks
# to this line don't invalidate the deps cache.
ENV PYTHONPATH=/app

# Mount point for the network volume holding HF model weights
RUN mkdir -p /opt/hf_cache

EXPOSE 8000

# Default: FastAPI on :8000 (Pod mode). RunPod Serverless overrides this
# entrypoint, but if you set RUNPOD_SERVERLESS=1 the same handler.py
# starts in serverless-handler mode.
CMD ["python", "-u", "/app/gpu_worker/handler.py"]
