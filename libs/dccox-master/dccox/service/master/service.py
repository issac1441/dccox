"""Business logic layer for the DC-Cox master.

Single Responsibility: Domain validation, state transitions, and orchestration.
Depends on the repository for data access (Dependency Inversion).
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from dccox.service.master.repository import ProjectRepository
from dccox.service.master.schemas import ProjectSchema, ProjectStatus, WorkerInfo
from dccox.usecase import Horizontal

logger = logging.getLogger(__name__)


class ProjectService:
    """Coordinates project lifecycle, worker registration, and analysis pipeline."""

    def __init__(self, repository: ProjectRepository) -> None:
        self._repo = repository

    # ── Project Management ──────────────────────────────────────────────

    def create_project(self, config: ProjectSchema) -> str:
        """Create a new project. Return the assigned project ID."""
        project_id = uuid.uuid4().hex[:8]
        self._repo.create(project_id, config)
        self._repo.add_event(
            project_id, f"Project '{config.name}' created. Waiting for workers."
        )
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        """Retrieve project details."""
        return self._repo.get(project_id)

    def list_projects(self) -> list[dict]:
        """List all projects."""
        return self._repo.list_all()

    # ── Worker Registration ─────────────────────────────────────────────

    def join_project(self, project_id: str, worker_name: str, n_features: int) -> str:
        """Register a worker in a project. Return the assigned worker_id.

        Raises
        ------
        ValueError
            If the feature count mismatches existing workers.
        """
        project = self._repo.get(project_id)
        if project is None:
            msg = f"Project '{project_id}' not found"
            raise ValueError(msg)

        current_n = project["n_features"]
        if current_n is None:
            self._repo.set_n_features(project_id, n_features)
        elif current_n != n_features:
            msg = f"Feature count mismatch: expected {current_n}, got {n_features}"
            raise ValueError(msg)

        worker_id = f"worker_{len(project['workers']) + 1}"
        self._repo.add_worker(
            project_id,
            WorkerInfo(
                worker_id=worker_id,
                worker_name=worker_name,
                n_features=n_features,
            ),
        )
        self._repo.add_event(
            project_id,
            f"Worker '{worker_name}' joined as '{worker_id}' (n_features: {n_features}).",
        )
        return worker_id

    # ── Pipeline: Lock ──────────────────────────────────────────────────

    def lock_project(self, project_id: str) -> None:
        """Lock the project to prevent further workers from joining.

        Raises
        ------
        ValueError
            If the project is not in JOINING state.
        """
        project = self._repo.get(project_id)
        if project is None:
            msg = f"Project '{project_id}' not found"
            raise ValueError(msg)
        if project["status"] != ProjectStatus.JOINING:
            msg = f"Cannot lock project in '{project['status']}' state"
            raise ValueError(msg)

        self._repo.update_status(project_id, ProjectStatus.LOCKED)
        self._repo.add_event(
            project_id,
            f"Project locked. Total workers: {len(project['workers'])}. Waiting to start.",
        )

    # ── Pipeline: Start ─────────────────────────────────────────────────

    def start_project(self, project_id: str) -> None:
        """Start analysis by generating the Xanc anchor matrix."""
        project = self._repo.get(project_id)
        if project is None:
            msg = f"Project '{project_id}' not found"
            raise ValueError(msg)

        config: ProjectSchema = project["config"]
        n_features = project["n_features"]

        uc = Horizontal()
        xanc = uc.global_create_Xanc(n_features, r=config.r)

        self._repo.store_xanc(project_id, xanc)
        self._repo.update_status(project_id, ProjectStatus.COMPUTING)
        self._repo.add_event(
            project_id,
            f"Analysis started. Generated Xanc matrix (shape: {xanc.shape}).",
        )

    def get_xanc(self, project_id: str) -> np.ndarray | None:
        """Retrieve the Xanc matrix."""
        return self._repo.get_xanc(project_id)

    # ── Pipeline: Proxy Data ────────────────────────────────────────────

    def submit_proxy(
        self,
        project_id: str,
        worker_id: str,
        x_tilde: list[list[float]] | None,
        xanc_tilde: list[list[float]] | None,
        y: list[list[float]],
        feature_sum: list[float],
    ) -> bool:
        """Store proxy data from a worker. Return True if all workers have submitted."""
        data = {
            "x_tilde": x_tilde,
            "xanc_tilde": xanc_tilde,
            "y": y,
            "feature_sum": feature_sum,
        }
        n_submitted = self._repo.store_proxy(project_id, worker_id, data)
        self._repo.mark_worker_submitted(project_id, worker_id)

        project = self._repo.get(project_id)
        n_workers = len(project["workers"])

        self._repo.add_event(
            project_id,
            f"Worker '{worker_id}' submitted proxy data ({n_submitted}/{n_workers}).",
        )

        if n_submitted >= n_workers:
            self._run_global_fit(project_id)
            return True
        return False

    # ── Pipeline: Global Fit ────────────────────────────────────────────

    def _run_global_fit(self, project_id: str) -> None:
        """Run the global model fitting once all proxy data is collected."""
        project = self._repo.get(project_id)
        config: ProjectSchema = project["config"]
        worker_ids = [w.worker_id for w in project["workers"]]
        proxy_snapshot = self._repo.get_all_proxy(project_id, worker_ids)

        self._repo.add_event(
            project_id, "All proxy data received. Initiating global fit..."
        )

        try:
            Xs_tilde, Xancs_tilde, ys, sums = _deserialize_proxy_data(
                proxy_snapshot, worker_ids
            )

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

            results = {
                wid: {
                    "coef": coef[i][0].tolist(),
                    "coef_var": coef_var[i][0].tolist(),
                    "baseline_hazard": baseline_hazard.to_dict(),
                    "feature_mean": feature_mean.tolist(),
                }
                for i, wid in enumerate(worker_ids)
            }
            self._repo.store_results(project_id, results)
            self._repo.update_status(project_id, ProjectStatus.COMPLETED)
            self._repo.add_event(project_id, "Global fit completed successfully.")

        except Exception as e:
            self._repo.update_status(project_id, ProjectStatus.FAILED)
            self._repo.set_error(project_id, "global_fit_model failed")
            self._repo.add_event(project_id, f"Global fit failed: {e}")
            logger.exception("Project %s: global fit failed", project_id)
            raise

    # ── Pipeline: Results ───────────────────────────────────────────────

    def get_worker_results(self, project_id: str, worker_id: str) -> dict | None:
        """Retrieve results for a specific worker."""
        return self._repo.get_worker_result(project_id, worker_id)

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Retrieve the event log for a project."""
        return self._repo.get_events(project_id)


def _deserialize_proxy_data(
    proxy_snapshot: dict[str, dict], worker_ids: list[str]
) -> tuple[list, list, list[np.ndarray], list[np.ndarray]]:
    """Convert JSON proxy data back to numpy arrays for the global fit."""
    Xs_tilde: list[list[np.ndarray | None]] = []
    Xancs_tilde: list[list[np.ndarray | None]] = []
    ys: list[np.ndarray] = []
    sums: list[np.ndarray] = []

    for wid in worker_ids:
        proxy = proxy_snapshot[wid]
        xt = np.array(proxy["x_tilde"]) if proxy["x_tilde"] is not None else None
        xat = np.array(proxy["xanc_tilde"]) if proxy["xanc_tilde"] is not None else None
        Xs_tilde.append([xt])
        Xancs_tilde.append([xat])
        ys.append(np.array(proxy["y"]))
        sums.append(np.array(proxy["feature_sum"]))

    return Xs_tilde, Xancs_tilde, ys, sums
