"""HTTP route handlers for the DC-Cox master API.

Single Responsibility: Thin HTTP layer — validates requests, delegates to
the service layer, and formats responses.  No business logic lives here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

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
from dccox.service.master.service import ProjectService

logger = logging.getLogger(__name__)


def create_router(service: ProjectService) -> APIRouter:
    """Build an APIRouter wired to the given *service* instance."""
    router = APIRouter(prefix="/api")

    # ── Health ──────────────────────────────────────────────────────────

    @router.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "dccox-master"}

    # ── Projects ────────────────────────────────────────────────────────

    @router.post("/projects", response_model=ProjectCreateResponse)
    async def create_project(config: ProjectSchema) -> ProjectCreateResponse:
        project_id = service.create_project(config)
        logger.info("Created project %s: %s", project_id, config.name)
        return ProjectCreateResponse(
            id=project_id,
            name=config.name,
            status=ProjectStatus.JOINING,
        )

    @router.get("/projects", response_model=list[ProjectDetail])
    async def list_projects() -> list[ProjectDetail]:
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
            for p in service.list_projects()
        ]

    @router.get("/projects/{project_id}", response_model=ProjectDetail)
    async def get_project(project_id: str) -> ProjectDetail:
        project = service.get_project(project_id)
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

    # ── Worker Registration ─────────────────────────────────────────────

    @router.post(
        "/projects/{project_id}/join",
        response_model=WorkerJoinResponse,
    )
    async def join_project(
        project_id: str, body: WorkerJoinRequest
    ) -> WorkerJoinResponse:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.JOINING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot join project in '{project['status']}' status",
            )

        try:
            worker_id = service.join_project(
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
        updated = service.get_project(project_id)
        total_workers = len(updated["workers"]) if updated else 0
        return WorkerJoinResponse(worker_id=worker_id, total_workers=total_workers)

    # ── Pipeline: Lock ──────────────────────────────────────────────────

    @router.post("/projects/{project_id}/lock")
    async def lock_project(project_id: str) -> dict[str, str]:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        try:
            service.lock_project(project_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "locked", "project_id": project_id}

    # ── Pipeline: Start ─────────────────────────────────────────────────

    @router.post("/projects/{project_id}/start")
    async def start_project(project_id: str) -> dict[str, str]:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.LOCKED:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot start project in '{project['status']}' status. "
                    "Must be locked first."
                ),
            )
        if len(project["workers"]) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one worker must join before starting",
            )
        service.start_project(project_id)
        return {"status": "started", "project_id": project_id}

    # ── Pipeline: Xanc ──────────────────────────────────────────────────

    @router.get("/projects/{project_id}/xanc")
    async def get_xanc(project_id: str) -> dict:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        xanc = service.get_xanc(project_id)
        if xanc is None:
            raise HTTPException(
                status_code=400,
                detail="Project not started yet — Xanc not available",
            )
        return {"xanc": xanc.tolist()}

    # ── Pipeline: Proxy Data ────────────────────────────────────────────

    @router.post("/projects/{project_id}/proxy/{worker_id}")
    async def submit_proxy(
        project_id: str, worker_id: str, body: ProxyDataPayload
    ) -> dict[str, str | bool]:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] not in (
            ProjectStatus.COMPUTING,
            ProjectStatus.COMPLETED,
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot submit proxy in '{project['status']}' status",
            )

        known_ids = {w.worker_id for w in project["workers"]}
        if worker_id not in known_ids:
            raise HTTPException(
                status_code=404, detail=f"Worker '{worker_id}' not found"
            )

        try:
            all_submitted = service.submit_proxy(
                project_id,
                worker_id,
                body.x_tilde,
                body.xanc_tilde,
                body.y,
                body.feature_sum,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Processing failed") from e

        return {
            "status": "submitted",
            "all_submitted": all_submitted,
            "project_id": project_id,
        }

    # ── Pipeline: Results ───────────────────────────────────────────────

    @router.get(
        "/projects/{project_id}/results/{worker_id}",
        response_model=WorkerResultResponse,
    )
    async def get_results(project_id: str, worker_id: str) -> WorkerResultResponse:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project["status"] != ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Results not ready (status: '{project['status']}')",
            )
        results = service.get_worker_results(project_id, worker_id)
        if results is None:
            raise HTTPException(status_code=404, detail=f"No results for '{worker_id}'")
        return WorkerResultResponse(**results)

    # ── Pipeline: Events ────────────────────────────────────────────────

    @router.get("/projects/{project_id}/events")
    async def get_events(project_id: str) -> list[dict[str, str]]:
        project = service.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return service.get_events(project_id)

    return router
