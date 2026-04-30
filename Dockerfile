FROM python:3.11-slim

# Install uv from the official image — no curl/apt required.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (cached layer — only rebuilt on pyproject/lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra auth --extra suggestions

# Copy application code.
COPY . .

# Railway injects $PORT at runtime; default to 8000 for local docker runs.
CMD ["sh", "-c", "uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
