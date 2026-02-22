# Use a multi-stage build to keep the final image small
FROM python:3.11-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy project
COPY . .

# Install worker's dependencies
RUN uv sync --frozen --package dccox-worker --package dccox-master

# -------------------------------------------------------

# Main stage
FROM python:3.11-slim-bookworm

# Copy uv and virtual environment
COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV DCCOX_WORKER_PORT=8001
ENV DCCOX_MASTER_URL=http://localhost:8000
ENV DCCOX_WORKER_NAME=my-worker

WORKDIR /app

# Copy application code
COPY libs/dccox-worker/ /app/libs/dccox-worker/
COPY libs/dccox-master/ /app/libs/dccox-master/
COPY dccox/ /app/dccox/

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the worker port (default 8001)
EXPOSE $DCCOX_WORKER_PORT

# Run the application
CMD ["uv", "run", "python", "-m", "dccox.service.worker.ui"]
