"""HTTP client for the DC-Cox master API.

Single Responsibility: Encapsulates all HTTP communication with the master
service. No business logic, no polling, no result caching.
"""

from __future__ import annotations

import contextlib
import logging

import httpx

logger = logging.getLogger(__name__)


class MasterClient:
    """Thin HTTP client for the DC-Cox master REST API."""

    def __init__(self, master_url: str, *, timeout: float = 120) -> None:
        self._base_url = master_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout)

    @property
    def base_url(self) -> str:
        """Return the master service base URL."""
        return self._base_url

    # ── Projects ────────────────────────────────────────────────────────

    def create_project(self, config: dict) -> str:
        """Create a project on the master. Return the project ID."""
        resp = self._http.post("/api/projects", json=config)
        resp.raise_for_status()
        project_id = resp.json()["id"]
        logger.info("Created project %s", project_id)
        return project_id

    def list_projects(self) -> list[dict]:
        """List all projects on the master."""
        resp = self._http.get("/api/projects")
        resp.raise_for_status()
        return resp.json()

    def get_project(self, project_id: str) -> dict:
        """Get project details."""
        resp = self._http.get(f"/api/projects/{project_id}")
        resp.raise_for_status()
        return resp.json()

    # ── Worker Registration ─────────────────────────────────────────────

    def join_project(self, project_id: str, worker_name: str, n_features: int) -> str:
        """Join a project. Return the assigned worker_id."""
        resp = self._http.post(
            f"/api/projects/{project_id}/join",
            json={"worker_name": worker_name, "n_features": n_features},
        )
        resp.raise_for_status()
        worker_id = resp.json()["worker_id"]
        logger.info("Joined as %s", worker_id)
        return worker_id

    # ── Pipeline Control ────────────────────────────────────────────────

    def lock_project(self, project_id: str) -> None:
        """Lock the project on the master."""
        resp = self._http.post(f"/api/projects/{project_id}/lock")
        resp.raise_for_status()
        logger.info("Project %s locked", project_id)

    def start_project(self, project_id: str) -> None:
        """Trigger analysis start on the master."""
        resp = self._http.post(f"/api/projects/{project_id}/start")
        resp.raise_for_status()
        logger.info("Project %s started", project_id)

    # ── Data Exchange ───────────────────────────────────────────────────

    def get_xanc(self, project_id: str) -> httpx.Response:
        """Fetch the Xanc matrix. Returns raw response for status checking."""
        return self._http.get(f"/api/projects/{project_id}/xanc")

    def submit_proxy(self, project_id: str, worker_id: str, payload: dict) -> None:
        """Submit proxy data to the master."""
        resp = self._http.post(
            f"/api/projects/{project_id}/proxy/{worker_id}",
            json=payload,
        )
        resp.raise_for_status()
        logger.info("Proxy data submitted")

    def get_results(self, project_id: str, worker_id: str) -> httpx.Response:
        """Fetch per-worker results. Returns raw response for status checking."""
        return self._http.get(f"/api/projects/{project_id}/results/{worker_id}")

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Fetch the event log from the master."""
        resp = self._http.get(f"/api/projects/{project_id}/events")
        resp.raise_for_status()
        return resp.json()

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the HTTP session."""
        self._http.close()

    def __del__(self) -> None:
        """Close the HTTP session on garbage collection."""
        with contextlib.suppress(Exception):
            self.close()
