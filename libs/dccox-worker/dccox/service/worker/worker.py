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
        self._results: dict[str, SurvivalFunction] = {}
        self._feature_names: dict[str, list[str]] = {}
        self._worker_id: str | None = None

    @property
    def worker_id(self) -> str | None:
        """Return the worker ID assigned after joining a project."""
        return self._worker_id

    def get_survival_function(self, project_id: str) -> SurvivalFunction | None:
        """Return the cached survival function for a project."""
        return self._results.get(project_id)

    def get_feature_names(self, project_id: str) -> list[str] | None:
        """Return the ordered feature names used for predictions."""
        names = self._feature_names.get(project_id)
        if names is None:
            surv = self._results.get(project_id)
            if surv is not None:
                names = [str(idx) for idx in surv.summary.index]
                self._feature_names[project_id] = names
        return names

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

        surv_func: SurvivalFunction = self._uc.local_recover_survival(
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
        self._results[project_id] = surv_func
        self._feature_names[project_id] = (
            list(keep_feature_cols) if keep_feature_cols else []
        )
        logger.info("Results stored for project %s", project_id)
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

    def get_worker_results(self, project_id: str) -> dict[str, str] | pd.DataFrame:
        """Return only the coefficients summary for compatibility."""
        tables = self.get_result_tables(project_id)
        if isinstance(tables, dict) and "error" in tables:
            return tables
        return tables["summary"]

    def get_result_tables(
        self, project_id: str
    ) -> dict[str, pd.DataFrame] | dict[str, str]:
        """Return formatted summary and baseline tables for a project."""
        surv = self._results.get(project_id)
        if surv is None:
            return {"error": f"No results found for project {project_id}"}
        return build_result_tables(surv)

    def close(self) -> None:
        """Close the HTTP session."""
        self._http.close()


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.reset_index()
    summary.rename(columns={summary.columns[0]: "feature"}, inplace=True)
    summary["feature"] = summary["feature"].apply(_format_feature)
    return summary


def _format_baseline_table(table: pd.DataFrame, value_label: str) -> pd.DataFrame:
    formatted = table.reset_index()
    time_col = formatted.columns[0]
    formatted.rename(columns={time_col: "timeline"}, inplace=True)
    value_cols = [col for col in formatted.columns if col != "timeline"]
    if value_cols:
        formatted.rename(columns={value_cols[0]: value_label}, inplace=True)
    return formatted


def build_result_tables(
    surv: SurvivalFunction,
) -> dict[str, pd.DataFrame]:
    """Build formatted summary and baseline tables."""
    return {
        "summary": _format_summary(surv.summary),
        "baseline_cumhazards": _format_baseline_table(
            surv.baseline_cumhazards, "baseline cumhazards"
        ),
        "baseline_survival": _format_baseline_table(
            surv.baseline_survival, "baseline survival"
        ),
        "baseline_hazard": _format_baseline_table(
            surv.baseline_hazard, "baseline hazard"
        ),
    }


def _format_feature(value: pd.Index | tuple[str] | list[str]) -> str:
    if isinstance(value, pd.Index):
        parts = value.tolist()
    elif isinstance(value, (tuple, list)):
        parts = value
    else:
        return str(value)
    return " + ".join(str(part) for part in parts)
