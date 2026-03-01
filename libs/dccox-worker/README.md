# dccox-worker

Worker node for DC-Cox federated Cox PH regression, with a Gradio web UI.

Connects to a `dccox-master` instance, loads local clinical data, computes proxy data, submits it for global fitting, and recovers the survival function — all through an interactive browser interface.

## Run Locally

From the **project root**:

```bash
# Install all workspace packages (first time only)
uv sync --all-packages

# Start worker on default port 8001
uv run python -m dccox.service.worker.ui
```

Or via uvicorn directly:

```bash
uv run uvicorn dccox.service.worker.ui:app --host 0.0.0.0 --port 8001
```

Make sure a master is already running (default: `http://localhost:8000`).

## Run with Docker

Build and run from the **project root** (the Dockerfile expects root as build context):

```bash
# Build
docker build -f libs/dccox-worker/Dockerfile -t dccox-worker .

# Run (standalone — master must be reachable)
docker run --rm \
  -p 8001:8001 \
  -e DCCOX_WORKER_PORT=8001 \
  -e DCCOX_MASTER_URL=http://host.docker.internal:8000 \
  -e DCCOX_WORKER_NAME=worker1 \
  dccox-worker
```

With a Docker network (recommended):

```bash
docker network create dccox-net

docker run --rm --name worker1 \
  --network dccox-net \
  -p 8001:8001 \
  -e DCCOX_WORKER_PORT=8001 \
  -e DCCOX_MASTER_URL=http://master:8000 \
  -e DCCOX_WORKER_NAME=worker1 \
  -v ./data:/mnt \
  dccox-worker
```

Open `http://localhost:8001` to access the UI.

> Mount your local data directory with `-v` so the worker can read clinical CSV files (e.g. `-v ./data:/mnt`). Then use `/mnt/your-file.csv` as the data path in the UI.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DCCOX_WORKER_PORT` | `8001` | Port the worker UI listens on |
| `DCCOX_MASTER_URL` | `http://localhost:8000` | Master service URL |
| `DCCOX_WORKER_NAME` | `my-worker` | Human-readable name for this worker |

Variables can also be set via a `.env` file (prefix: `DCCOX_`).

## Module Structure

```
dccox/service/worker/
├── config.py             # Pydantic settings (env-based configuration)
├── client.py             # HTTP client for the master REST API
├── pipeline.py           # Local pipeline orchestration (poll, compute, submit)
├── result_formatter.py   # Result table formatting utilities
├── worker.py             # Facade coordinating client, pipeline, and results
└── ui/                   # Gradio web interface
    ├── __init__.py       # Package re-exports (app, create_app, create_ui)
    ├── __main__.py       # Entry point for `python -m dccox.service.worker.ui`
    ├── app.py            # Gradio layout + handler wiring + FastAPI mount
    ├── theme.py          # CSS and Gradio theme configuration
    ├── prediction.py     # Prediction method specs (frozen dataclasses)
    ├── helpers.py        # Shared utilities (temp files, plots, formatting)
    └── handlers.py       # Gradio event handler business logic
```
