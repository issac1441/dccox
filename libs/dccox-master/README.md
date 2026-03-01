# dccox-master

Central orchestration server for DC-Cox federated Cox PH regression.

Manages project lifecycle, worker registration, proxy data collection, and global model fitting via a REST API built on FastAPI.

## Run Locally

From the **project root**:

```bash
# Install all workspace packages (first time only)
uv sync --all-packages

# Start master on default port 8000
uv run uvicorn dccox.service.master.app:app --host 0.0.0.0 --port 8000
```

Or run directly as a module:

```bash
uv run python -m dccox.service.master.app
```

## Run with Docker

Build and run from the **project root** (the Dockerfile expects root as build context):

```bash
# Build
docker build -f libs/dccox-master/Dockerfile -t dccox-master .

# Run
docker run --rm \
  -e DCCOX_MASTER_PORT=8000 \
  -p 8000:8000 \
  dccox-master
```

With a Docker network (for worker connectivity):

```bash
docker network create dccox-net

docker run --rm --name master \
  --network dccox-net \
  -e DCCOX_MASTER_PORT=8000 \
  dccox-master
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DCCOX_MASTER_PORT` | `8000` | Port the master listens on |

Variables can also be set via a `.env` file (prefix: `DCCOX_`).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/projects` | Create a new project |
| `GET` | `/api/projects` | List all projects |
| `GET` | `/api/projects/{id}` | Get project details |
| `POST` | `/api/projects/{id}/join` | Register a worker |
| `POST` | `/api/projects/{id}/lock` | Lock project (stop accepting workers) |
| `POST` | `/api/projects/{id}/start` | Start analysis (generate Xanc) |
| `GET` | `/api/projects/{id}/xanc` | Get the anchor matrix |
| `POST` | `/api/projects/{id}/proxy/{wid}` | Submit proxy data |
| `GET` | `/api/projects/{id}/results/{wid}` | Get per-worker results |
| `GET` | `/api/projects/{id}/events` | Get event log |

Interactive API docs available at `/docs` (Swagger UI) when running.

## Module Structure

```
dccox/service/master/
├── app.py           # FastAPI app factory — wires repository, service, router
├── config.py        # Pydantic settings (env-based configuration)
├── schemas.py       # Request/response Pydantic models
├── repository.py    # Thread-safe in-memory data access (CRUD)
├── service.py       # Business logic and orchestration
└── routes.py        # HTTP route handlers (APIRouter)
```
