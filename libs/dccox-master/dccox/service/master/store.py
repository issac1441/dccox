"""In-memory project store for the DC-Cox master."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
import uuid

import numpy as np

from dccox.service.master.schemas import ProjectSchema, ProjectStatus, WorkerInfo
from dccox.usecase import Horizontal

logger = logging.getLogger(__name__)


class ProjectStore:
    """Manage project lifecycle, proxy data collection, and global operations."""

    def __init__(self) -> None:
        self._projects: dict[str, dict] = {}
        self._xanc: dict[str, np.ndarray] = {}
        self._proxy_data: dict[str, dict[str, dict]] = {}
        self._results: dict[str, dict] = {}
        self._events: dict[str, list[dict[str, str]]] = {}

    def add_event(self, project_id: str, message: str) -> None:
        """Add a timestamped event to the project's event log."""
        if project_id not in self._events:
            self._events[project_id] = []

        event = {
            "time": datetime.now(tz=UTC).isoformat(),
            "message": message,
        }
        self._events[project_id].append(event)
        logger.info("[Project %s] %s", project_id, message)

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Get the event log for a project."""
        return self._events.get(project_id, [])

    # ── Project Management ─────────────────────────────────────────────

    def create_project(self, config: ProjectSchema) -> str:
        """Create a new project and return its ID."""
        project_id = uuid.uuid4().hex[:8]
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
        self.add_event(
            project_id, f"Project '{config.name}' created. Waiting for workers."
        )
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        """Get project details by ID."""
        return self._projects.get(project_id)

    def list_projects(self) -> list[dict]:
        """List all projects."""
        return list(self._projects.values())

    # ── Worker Registration ────────────────────────────────────────────

    def join_project(self, project_id: str, worker_name: str, n_features: int) -> str:
        """Register a worker in a project. Return the assigned worker_id."""
        project = self._projects[project_id]

        # Validate n_features consistency
        if project["n_features"] is None:
            project["n_features"] = n_features
        elif project["n_features"] != n_features:
            msg = (
                f"Feature count mismatch: expected {project['n_features']}, "
                f"got {n_features}"
            )
            raise ValueError(msg)

        worker_id = f"worker_{len(project['workers']) + 1}"
        project["workers"].append(
            WorkerInfo(
                worker_id=worker_id,
                worker_name=worker_name,
                n_features=n_features,
            )
        )
        self.add_event(
            project_id,
            f"Worker '{worker_name}' joined as '{worker_id}' (n_features: {n_features}).",
        )
        return worker_id

    # ── Pipeline: Start ────────────────────────────────────────────────

    def lock_project(self, project_id: str) -> None:
        """Lock the project to prevent further workers from joining."""
        project = self._projects[project_id]
        if project["status"] != ProjectStatus.JOINING:
            raise ValueError(f"Cannot lock project in '{project['status']}' state")

        project["status"] = ProjectStatus.LOCKED
        self.add_event(
            project_id,
            f"Project locked. Total workers: {len(project['workers'])}. Waiting to start.",
        )

    def start_project(self, project_id: str) -> None:
        """Start analysis: generate Xanc."""
        project = self._projects[project_id]
        config: ProjectSchema = project["config"]
        n_features = project["n_features"]

        uc = Horizontal()
        xanc = uc.global_create_Xanc(n_features, r=config.r)
        self._xanc[project_id] = xanc
        project["status"] = ProjectStatus.COMPUTING
        self.add_event(
            project_id,
            f"Analysis started. Generated Xanc matrix (shape: {xanc.shape}).",
        )

    def get_xanc(self, project_id: str) -> np.ndarray | None:
        """Get the Xanc matrix for a project."""
        return self._xanc.get(project_id)

    # ── Pipeline: Proxy Data ───────────────────────────────────────────

    def submit_proxy(
        self,
        project_id: str,
        worker_id: str,
        x_tilde: list[list[float]] | None,
        xanc_tilde: list[list[float]] | None,
        y: list[list[float]],
        feature_sum: list[float],
    ) -> bool:
        """Store proxy data from a worker. Return True if all workers submitted."""
        self._proxy_data[project_id][worker_id] = {
            "x_tilde": x_tilde,
            "xanc_tilde": xanc_tilde,
            "y": y,
            "feature_sum": feature_sum,
        }

        # Mark worker as submitted
        project = self._projects[project_id]
        for w in project["workers"]:
            if w.worker_id == worker_id:
                w.has_submitted_proxy = True

        n_workers = len(project["workers"])
        n_submitted = len(self._proxy_data[project_id])
        self.add_event(
            project_id,
            f"Worker '{worker_id}' submitted proxy data ({n_submitted}/{n_workers}).",
        )

        if n_submitted >= n_workers:
            self._run_global_fit(project_id)
            return True
        return False

    # ── Pipeline: Global Fit ───────────────────────────────────────────

    def _run_global_fit(self, project_id: str) -> None:
        """Run global_fit_model once all proxy data is collected."""
        project = self._projects[project_id]
        config: ProjectSchema = project["config"]
        worker_ids = [w.worker_id for w in project["workers"]]

        self.add_event(project_id, "All proxy data received. Initiating global fit...")

        try:
            # Reconstruct arrays from JSON
            Xs_tilde: list[list[np.ndarray | None]] = []
            Xancs_tilde: list[list[np.ndarray | None]] = []
            ys: list[np.ndarray] = []
            sums: list[np.ndarray] = []

            for wid in worker_ids:
                proxy = self._proxy_data[project_id][wid]
                xt = (
                    np.array(proxy["x_tilde"]) if proxy["x_tilde"] is not None else None
                )
                xat = (
                    np.array(proxy["xanc_tilde"])
                    if proxy["xanc_tilde"] is not None
                    else None
                )
                Xs_tilde.append([xt])
                Xancs_tilde.append([xat])
                ys.append(np.array(proxy["y"]))
                sums.append(np.array(proxy["feature_sum"]))

            uc = Horizontal()
            coef, coef_var, baseline_hazard, feature_mean = uc.global_fit_model(
                Xs_tilde,
                Xancs_tilde,
                ys,
                sums,
                alpha=config.alpha,
                step_size=config.step_size,
                var_thres=config.var_thres,
                m_hat=config.m_hat,
            )

            # Store per-worker results
            self._results[project_id] = {}
            for i, wid in enumerate(worker_ids):
                self._results[project_id][wid] = {
                    "coef": coef[i][0].tolist(),
                    "coef_var": coef_var[i][0].tolist(),
                    "baseline_hazard": baseline_hazard.to_dict(),
                    "feature_mean": feature_mean.tolist(),
                }

            project["status"] = ProjectStatus.COMPLETED
            self.add_event(project_id, "Global fit completed successfully.")

        except Exception as e:
            project["status"] = ProjectStatus.FAILED
            project["error"] = "global_fit_model failed"
            self.add_event(project_id, f"Global fit failed: {e}")
            logger.exception("Project %s: global fit failed", project_id)
            raise

    # ── Pipeline: Results ──────────────────────────────────────────────

    def get_worker_results(self, project_id: str, worker_id: str) -> dict | None:
        """Get results for a specific worker."""
        return self._results.get(project_id, {}).get(worker_id)


# Global singleton
store = ProjectStore()
