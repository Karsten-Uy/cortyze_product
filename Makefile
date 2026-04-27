# Cortyze backend convenience targets.
# Run from cortyze_product/.

REGISTRY ?= ghcr.io/$(GITHUB_USER)
IMAGE := cortyze-gpu-worker
TAG ?= latest
FULL := $(REGISTRY)/$(IMAGE):$(TAG)

.PHONY: help dev-up dev-down dev-status dev-logs api-run test image-build image-shell

help:
	@echo "Targets:"
	@echo "  dev-up        Start local MinIO (object storage)"
	@echo "  dev-down      Stop local MinIO"
	@echo "  dev-status    Show MinIO status"
	@echo "  dev-logs      Tail MinIO logs"
	@echo "  api-run       Run the FastAPI dev server (uvicorn --reload)"
	@echo "  test          Run pytest"
	@echo "  image-build   Build + push the GPU worker image (needs GITHUB_USER, Docker)"
	@echo "  image-shell   Open a shell in the built image to debug"

# Local dev infra
dev-up:
	./scripts/dev_minio.sh start

dev-down:
	./scripts/dev_minio.sh stop

dev-status:
	./scripts/dev_minio.sh status

dev-logs:
	./scripts/dev_minio.sh logs

# API
api-run:
	uv run uvicorn api.main:app --reload

test:
	uv run pytest -v

# GPU worker image
image-build:
	@if [ -z "$(GITHUB_USER)" ]; then \
	  echo "ERROR: set GITHUB_USER (e.g. GITHUB_USER=yourname make image-build)"; \
	  exit 1; \
	fi
	./docker/build.sh

image-shell:
	docker run --rm -it --platform linux/amd64 $(FULL) bash
