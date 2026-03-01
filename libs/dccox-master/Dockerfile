# Use a multi-stage build to keep the final image small
FROM python:3.11-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy project
COPY . .

# Install master's dependencies
RUN uv sync --frozen --package dccox-master

# -------------------------------------------------------

# Main stage
FROM python:3.11-slim-bookworm

# Copy uv and virtual environment
COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV DCCOX_MASTER_PORT=8000

WORKDIR /app

# Copy application code
COPY libs/dccox-master/ /app/libs/dccox-master/
COPY dccox/ /app/dccox/

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the master port (default 8000)
EXPOSE $DCCOX_MASTER_PORT

# Run the application
CMD ["uv", "run", "uvicorn", "dccox.service.master.app:app", "--host", "0.0.0.0"]
