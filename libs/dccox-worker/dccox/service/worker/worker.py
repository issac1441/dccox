"""DC-Cox federated worker library."""

from __future__ import annotations

import logging
import time

import httpx
import numpy as np
import pandas as pd

from dccox.cox import SurvivalFunction
from dccox.usecase import Horizontal

logger = logging.getLogger(__name__)


class DCCoxWorker:
    """Worker for the DC-Cox federated analysis protocol.

    Wraps HTTP calls to the master and local ``Horizontal`` operations.

    Parameters
    ----------
    master_url : str
        Base URL of the master service (e.g. ``http://localhost:8000``).
    """

    def __init__(self, master_url: str) -> None:
        self.master_url = master_url.rstrip("/")
        self._http = httpx.Client(base_url=self.master_url, timeout=120)
        self._uc = Horizontal()
        self._worker_id: str | None = None

    @property
    def worker_id(self) -> str | None:
        """Return the worker ID assigned after joining a project."""
        return self._worker_id

    # ── Project Management ─────────────────────────────────────────────

    def create_project(self, config: dict) -> str:
        """Create a project on the master. Return project ID."""
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

    # ── Join ───────────────────────────────────────────────────────────

    def join_project(self, project_id: str, worker_name: str, n_features: int) -> str:
        """Join a project. Return assigned worker_id."""
        resp = self._http.post(
            f"/api/projects/{project_id}/join",
            json={"worker_name": worker_name, "n_features": n_features},
        )
        resp.raise_for_status()
        self._worker_id = resp.json()["worker_id"]
        logger.info("Joined as %s", self._worker_id)
        return self._worker_id

    # ── Start ──────────────────────────────────────────────────────────

    def start_project(self, project_id: str) -> None:
        """Trigger analysis start on the master."""
        resp = self._http.post(f"/api/projects/{project_id}/start")
        resp.raise_for_status()
        logger.info("Project %s started", project_id)

    def lock_project(self, project_id: str) -> None:
        """Lock the project on the master."""
        resp = self._http.post(f"/api/projects/{project_id}/lock")
        resp.raise_for_status()
        logger.info("Project %s locked", project_id)

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Fetch the event log from the master."""
        resp = self._http.get(f"/api/projects/{project_id}/events")
        resp.raise_for_status()
        return resp.json()

    # ── Pipeline: local compute + submit ───────────────────────────────

    def run_local_pipeline(
        self,
        project_id: str,
        data_path: str,
        *,
        poll_interval: float = 2.0,
    ) -> SurvivalFunction:
        """Run the full worker-side pipeline.

        1. Wait for Xanc from master
        2. Load local data + create proxy data
        3. Submit proxy data
        4. Wait for global results
        5. Recover survival function

        Parameters
        ----------
        project_id : str
            Project ID.
        data_path : str
            Path to local clinical CSV file.
        poll_interval : float
            Seconds between polling attempts.

        Returns
        -------
        SurvivalFunction
            The recovered survival function for this worker.
        """
        if self._worker_id is None:
            msg = "Must join a project first"
            raise RuntimeError(msg)

        # Get project config
        project = self.get_project(project_id)
        config = project["config"]

        # 1. Wait for Xanc
        xanc = self._poll_xanc(project_id, poll_interval)

        # 2. Local computation
        X, y, keep_feature_cols, _meta = self._uc.local_load_metadata(
            data_path,
            keep_feature_cols=config.get("keep_feature_cols"),
            meta_cols=config.get("meta_cols"),
        )
        F, X_tilde, Xanc_tilde, feature_sum = self._uc.local_create_proxy_data(
            X,
            xanc,
            y,
            k=config.get("k", 20),
            bs_prop=config.get("bs_prop", 0.6),
            bs_times=config.get("bs_times", 20),
            bs_replace=config.get("bs_replace", False),
            alpha=config.get("alpha", 0.05),
            step_size=config.get("step_size", 0.5),
        )
        logger.info("Local proxy data computed (%d samples)", len(X))

        # 3. Submit proxy data
        payload = {
            "x_tilde": X_tilde[0].tolist() if X_tilde[0] is not None else None,
            "xanc_tilde": (
                Xanc_tilde[0].tolist() if Xanc_tilde[0] is not None else None
            ),
            "y": y.tolist(),
            "feature_sum": feature_sum.tolist(),
        }
        resp = self._http.post(
            f"/api/projects/{project_id}/proxy/{self._worker_id}",
            json=payload,
        )
        resp.raise_for_status()
        logger.info("Proxy data submitted")

        # 4. Wait for global results
        results = self._poll_results(project_id, poll_interval)

        # 5. Recover survival
        coef = np.array(results["coef"])
        coef_var = np.array(results["coef_var"])
        baseline_hazard = pd.DataFrame(results["baseline_hazard"])
        feature_mean = np.array(results["feature_mean"])

        surv_func = self._uc.local_recover_survival(
            keep_feature_cols,
            coef,
            coef_var,
            baseline_hazard,
            feature_mean,
            F,
            alpha=config.get("alpha", 0.05),
            centering=config.get("centering"),
        )
        logger.info("Survival function recovered")
        return surv_func

    # ── Polling helpers ────────────────────────────────────────────────

    def _poll_xanc(self, project_id: str, interval: float) -> np.ndarray:
        """Poll until Xanc is available."""
        while True:
            resp = self._http.get(f"/api/projects/{project_id}/xanc")
            if resp.status_code == 200:
                return np.array(resp.json()["xanc"])
            logger.debug("Waiting for Xanc...")
            time.sleep(interval)

    def _poll_results(self, project_id: str, interval: float) -> dict:
        """Poll until per-worker results are available."""
        while True:
            resp = self._http.get(
                f"/api/projects/{project_id}/results/{self._worker_id}"
            )
            if resp.status_code == 200:
                return resp.json()
            logger.debug("Waiting for global results...")
            time.sleep(interval)

    def get_worker_results(self, project_id: str, worker_id: str) -> dict:
        """Fetch global results for a specific worker from the master."""
        resp = self._http.get(f"/api/projects/{project_id}/results/{worker_id}")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP session."""
        self._http.close()
