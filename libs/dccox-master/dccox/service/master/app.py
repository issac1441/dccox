"""DC-Cox master — FastAPI application."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from dccox.service.master.schemas import (
    ProjectCreateResponse,
    ProjectDetail,
    ProjectSchema,
    ProjectStatus,
    ProxyDataPayload,
    WorkerJoinRequest,
    WorkerJoinResponse,
    WorkerResultResponse,
)
from dccox.service.master.store import store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("DC-Cox master starting up")
    yield
    logger.info("DC-Cox master shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DC-Cox Master",
        description="Federated Cox PH Regression — Master Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Health ──────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "dccox-master"}

    # ── Projects ───────────────────────────────────────────────────────

    @app.post("/api/projects", response_model=ProjectCreateResponse)
    async def create_project(config: ProjectSchema) -> ProjectCreateResponse:
        """Create a new analysis project."""
        project_id = store.create_project(config)
        logger.info("Created project %s: %s", project_id, config.name)
        return ProjectCreateResponse(
            id=project_id,
            name=config.name,
            status=ProjectStatus.CREATED,
        )

    @app.get("/api/projects", response_model=list[ProjectDetail])
    async def list_projects() -> list[ProjectDetail]:
        """List all projects."""
        projects = store.list_projects()
        return [
            ProjectDetail(
                id=p["id"],
                config=p["config"],
                status=p["status"],
                workers=p["workers"],
                n_features=p["n_features"],
                error=p["error"],
                created_at=p["created_at"],
            )
            for p in projects
        ]

    @app.get("/api/projects/{project_id}", response_model=ProjectDetail)
    async def get_project(project_id: str) -> ProjectDetail:
        """Get project details."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectDetail(
            id=project["id"],
            config=project["config"],
            status=project["status"],
            workers=project["workers"],
            n_features=project["n_features"],
            error=project["error"],
            created_at=project["created_at"],
        )

    # ── Worker Registration ────────────────────────────────────────────

    @app.post(
        "/api/projects/{project_id}/join",
        response_model=WorkerJoinResponse,
    )
    async def join_project(
        project_id: str, body: WorkerJoinRequest
    ) -> WorkerJoinResponse:
        """Join a project as a worker."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.CREATED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot join project in '{project['status']}' status",
            )

        try:
            worker_id = store.join_project(
                project_id, body.worker_name, body.n_features
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        logger.info(
            "Worker '%s' joined project %s as %s (n_features=%d)",
            body.worker_name,
            project_id,
            worker_id,
            body.n_features,
        )
        return WorkerJoinResponse(
            worker_id=worker_id,
            total_workers=len(project["workers"]),
        )

    # ── Pipeline: Start ────────────────────────────────────────────────

    @app.post("/api/projects/{project_id}/start")
    async def start_project(project_id: str) -> dict[str, str]:
        """Start the analysis (generates Xanc)."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.CREATED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot start project in '{project['status']}' status",
            )
        if len(project["workers"]) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one worker must join before starting",
            )

        store.start_project(project_id)
        return {"status": "started", "project_id": project_id}

    # ── Pipeline: Xanc ─────────────────────────────────────────────────

    @app.get("/api/projects/{project_id}/xanc")
    async def get_xanc(project_id: str) -> dict:
        """Get the global anchor matrix Xanc."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        xanc = store.get_xanc(project_id)
        if xanc is None:
            raise HTTPException(
                status_code=400,
                detail="Project not started yet — Xanc not available",
            )

        return {"xanc": xanc.tolist()}

    # ── Pipeline: Proxy Data ───────────────────────────────────────────

    @app.post("/api/projects/{project_id}/proxy/{worker_id}")
    async def submit_proxy(
        project_id: str, worker_id: str, body: ProxyDataPayload
    ) -> dict[str, str | bool]:
        """Submit proxy data from a worker."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] not in (ProjectStatus.STARTED, ProjectStatus.COMPLETED):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot submit proxy in '{project['status']}' status",
            )

        known_ids = {c.worker_id for c in project["workers"]}
        if worker_id not in known_ids:
            raise HTTPException(
                status_code=404, detail=f"Worker '{worker_id}' not found"
            )

        try:
            all_submitted = store.submit_proxy(
                project_id,
                worker_id,
                body.x_tilde,
                body.xanc_tilde,
                body.y,
                body.feature_sum,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Processing failed: {e}"
            ) from e

        return {
            "status": "submitted",
            "all_submitted": all_submitted,
            "project_id": project_id,
        }

    # ── Pipeline: Results ──────────────────────────────────────────────

    @app.get(
        "/api/projects/{project_id}/results/{worker_id}",
        response_model=WorkerResultResponse,
    )
    async def get_results(project_id: str, worker_id: str) -> WorkerResultResponse:
        """Get per-worker global results."""
        project = store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Results not ready (status: '{project['status']}')",
            )

        results = store.get_worker_results(project_id, worker_id)
        if results is None:
            raise HTTPException(status_code=404, detail=f"No results for '{worker_id}'")

        return WorkerResultResponse(**results)

    return app


# Module-level app for `uvicorn dccox.service.master.app:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "dccox.service.master.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
