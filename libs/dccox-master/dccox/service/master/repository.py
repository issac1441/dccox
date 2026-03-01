"""In-memory data access layer for the DC-Cox master.

Single Responsibility: Pure CRUD operations on project data with thread safety.
No business logic, no domain validation, no external dependencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from threading import RLock

import numpy as np

from dccox.service.master.schemas import ProjectSchema, ProjectStatus, WorkerInfo

logger = logging.getLogger(__name__)


class ProjectRepository:
    """Thread-safe in-memory storage for projects and their associated data."""

    def __init__(self) -> None:
        self._projects: dict[str, dict] = {}
        self._xanc: dict[str, np.ndarray] = {}
        self._proxy_data: dict[str, dict[str, dict]] = {}
        self._results: dict[str, dict] = {}
        self._events: dict[str, list[dict[str, str]]] = {}
        self._lock = RLock()

    # ── Project CRUD ────────────────────────────────────────────────────

    def create(self, project_id: str, config: ProjectSchema) -> None:
        """Insert a new project record."""
        with self._lock:
            self._projects[project_id] = {
                "id": project_id,
                "config": config,
                "status": ProjectStatus.JOINING,
                "workers": [],
                "n_features": None,
                "error": None,
                "created_at": datetime.now(tz=UTC).isoformat(),
            }
            self._proxy_data[project_id] = {}

    def get(self, project_id: str) -> dict | None:
        """Retrieve a project by ID."""
        with self._lock:
            return self._projects.get(project_id)

    def list_all(self) -> list[dict]:
        """Return all projects."""
        with self._lock:
            return list(self._projects.values())

    def update_status(self, project_id: str, status: ProjectStatus) -> None:
        """Set the project status."""
        with self._lock:
            self._projects[project_id]["status"] = status

    def set_error(self, project_id: str, error: str) -> None:
        """Set the project error message."""
        with self._lock:
            self._projects[project_id]["error"] = error

    def set_n_features(self, project_id: str, n_features: int) -> None:
        """Set the expected feature count for a project."""
        with self._lock:
            self._projects[project_id]["n_features"] = n_features

    # ── Workers ─────────────────────────────────────────────────────────

    def add_worker(self, project_id: str, worker: WorkerInfo) -> None:
        """Append a worker to the project's worker list."""
        with self._lock:
            self._projects[project_id]["workers"].append(worker)

    def mark_worker_submitted(self, project_id: str, worker_id: str) -> None:
        """Mark a worker's proxy data as submitted."""
        with self._lock:
            project = self._projects[project_id]
            for idx, w in enumerate(project["workers"]):
                if w.worker_id == worker_id:
                    project["workers"][idx] = w.model_copy(
                        update={"has_submitted_proxy": True}
                    )
                    break

    # ── Xanc ────────────────────────────────────────────────────────────

    def store_xanc(self, project_id: str, xanc: np.ndarray) -> None:
        """Store the generated Xanc matrix."""
        with self._lock:
            self._xanc[project_id] = xanc

    def get_xanc(self, project_id: str) -> np.ndarray | None:
        """Retrieve the Xanc matrix for a project."""
        with self._lock:
            return self._xanc.get(project_id)

    # ── Proxy Data ──────────────────────────────────────────────────────

    def store_proxy(self, project_id: str, worker_id: str, data: dict) -> int:
        """Store proxy data for a worker. Return total proxy submissions count."""
        with self._lock:
            self._proxy_data[project_id][worker_id] = data
            return len(self._proxy_data[project_id])

    def get_all_proxy(self, project_id: str, worker_ids: list[str]) -> dict[str, dict]:
        """Return a snapshot of proxy data for the given workers."""
        with self._lock:
            return {wid: self._proxy_data[project_id][wid].copy() for wid in worker_ids}

    # ── Results ─────────────────────────────────────────────────────────

    def store_results(self, project_id: str, results: dict[str, dict]) -> None:
        """Store per-worker results for a project."""
        with self._lock:
            self._results[project_id] = results

    def get_worker_result(self, project_id: str, worker_id: str) -> dict | None:
        """Retrieve results for a specific worker."""
        with self._lock:
            result = self._results.get(project_id, {}).get(worker_id)
            return result.copy() if result is not None else None

    # ── Events ──────────────────────────────────────────────────────────

    def add_event(self, project_id: str, message: str) -> None:
        """Append a timestamped event to the project's log."""
        with self._lock:
            if project_id not in self._events:
                self._events[project_id] = []
            self._events[project_id].append(
                {
                    "time": datetime.now(tz=UTC).isoformat(),
                    "message": message,
                }
            )
        logger.info("[Project %s] %s", project_id, message)

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Return the event log for a project."""
        with self._lock:
            return list(self._events.get(project_id, []))
