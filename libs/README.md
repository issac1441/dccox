# DC-COX Services

This directory contains the two service packages that make up the DC-Cox federated analysis system.

| Package | Description | Default Port |
|---|---|---|
| [dccox-master](./dccox-master/) | Central orchestration server (FastAPI) | 8000 |
| [dccox-worker](./dccox-worker/) | Worker node with Gradio UI | 8001 |

Both packages are part of the [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) defined in the root `pyproject.toml` and share the core `dccox` library via namespace packaging.

## Quick Start (Local)

From the **project root**:

```bash
# Install all workspace packages
uv sync --all-packages

# Terminal 1 — start master
uv run uvicorn dccox.service.master.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — start worker
uv run python -m dccox.service.worker.ui
```

## Quick Start (Docker)

Build images from the **project root** (the Dockerfiles use root as build context):

```bash
# Build
docker build -f libs/dccox-master/Dockerfile -t dccox-master .
docker build -f libs/dccox-worker/Dockerfile -t dccox-worker .

# Create a shared network (for worker connectivity to master)
docker network create dccox-net

# Run master
docker run --rm --name master \
  --network dccox-net \
  -e DCCOX_MASTER_PORT=8000 \
  dccox-master

# Run worker (mount local data directory)
docker run --rm --name worker1 \
  --network dccox-net \
  -p 8001:8001 \
  -e DCCOX_WORKER_PORT=8001 \
  -e DCCOX_MASTER_URL=http://master:8000 \
  -e DCCOX_WORKER_NAME=worker1 \
  -v ./data:/mnt \
  dccox-worker
```

Open `http://localhost:8001` to access the worker UI.

> [!NOTE]
> The master URL `http://master:8000` is configured by the `--name master` flag of the `docker run` command.

## Environment Variables

All environment variables use the `DCCOX_` prefix and can also be set via a `.env` file.

|Image| Variable | Default | Description |
|---|---|---|---|
| dccox-master | `DCCOX_MASTER_PORT` | `8000` | Master service port |
| dccox-worker | `DCCOX_WORKER_PORT` | `8001` | Worker service port |
| dccox-worker | `DCCOX_MASTER_URL` | `http://localhost:8000` | Master URL (used by worker) |
| dccox-worker | `DCCOX_WORKER_NAME` | `my-worker` | Human-readable worker name |

## Architecture

```
┌──────────────────┐        HTTP         ┌──────────────────┐
│   dccox-master   │◄───────────────────►│   dccox-worker   │
│   (FastAPI)      │   /api/projects/*   │   (Gradio UI)    │
│                  │                     │                  │
│  repository.py   │                     │  client.py       │
│  service.py      │                     │  pipeline.py     │
│  routes.py       │                     │  worker.py       │
│  schemas.py      │                     │  ui/             │
└──────────────────┘                     └──────────────────┘
        │                                        │
        └────────────┐          ┌────────────────┘
                     ▼          ▼
              ┌──────────────────────┐
              │    dccox (core)      │
              │  cox.py · usecase.py │
              └──────────────────────┘
```
