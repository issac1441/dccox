"""Local pipeline orchestration for the DC-Cox worker.

Single Responsibility: Coordinates the worker-side analysis pipeline
(poll Xanc, compute proxy data, submit, poll results, recover survival).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from dccox.cox import SurvivalFunction
from dccox.service.worker.client import MasterClient
from dccox.usecase import Horizontal

logger = logging.getLogger(__name__)


class LocalPipeline:
    """Executes the worker-side federated pipeline steps."""

    def __init__(self, client: MasterClient, usecase: Horizontal) -> None:
        self._client = client
        self._uc = usecase

    def run(
        self,
        project_id: str,
        worker_id: str,
        data_path: str,
        config: dict,
        *,
        poll_interval: float = 2.0,
        poll_timeout: float | None = 600.0,
    ) -> tuple[SurvivalFunction, list[str]]:
        """Run the full worker-side pipeline.

        Returns
        -------
        tuple[SurvivalFunction, list[str]]
            The recovered survival function and the ordered feature names.
        """
        xanc = self._poll_xanc(project_id, poll_interval, poll_timeout)

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

        payload = {
            "x_tilde": X_tilde[0].tolist() if X_tilde[0] is not None else None,
            "xanc_tilde": (
                Xanc_tilde[0].tolist() if Xanc_tilde[0] is not None else None
            ),
            "y": y.tolist(),
            "feature_sum": feature_sum.tolist(),
        }
        self._client.submit_proxy(project_id, worker_id, payload)

        results = self._poll_results(project_id, worker_id, poll_interval, poll_timeout)

        surv_func = self._recover_survival(results, keep_feature_cols, F, config)
        logger.info("Survival function recovered")

        feature_names = list(keep_feature_cols) if keep_feature_cols else []
        return surv_func, feature_names

    def _recover_survival(
        self,
        results: dict,
        keep_feature_cols: list[str] | None,
        F: np.ndarray,
        config: dict,
    ) -> SurvivalFunction:
        """Transform global results back to the original feature space."""
        coef = np.array(results["coef"])
        coef_var = np.array(results["coef_var"])
        baseline_hazard = pd.DataFrame(results["baseline_hazard"])
        feature_mean = np.array(results["feature_mean"])

        return self._uc.local_recover_survival(
            keep_feature_cols,
            coef,
            coef_var,
            baseline_hazard,
            feature_mean,
            F,
            alpha=config.get("alpha", 0.05),
            centering=config.get("centering"),
        )

    def _poll_xanc(
        self, project_id: str, interval: float, timeout: float | None
    ) -> np.ndarray:
        """Poll until Xanc is available."""
        start = time.monotonic()
        while True:
            resp = self._client.get_xanc(project_id)
            if resp.status_code == 200:
                return np.array(resp.json()["xanc"])
            logger.debug("Waiting for Xanc...")
            time.sleep(interval)
            if timeout is not None and time.monotonic() - start >= timeout:
                msg = f"Timed out waiting for Xanc for project {project_id}"
                raise TimeoutError(msg)

    def _poll_results(
        self,
        project_id: str,
        worker_id: str,
        interval: float,
        timeout: float | None,
    ) -> dict:
        """Poll until per-worker results are available."""
        start = time.monotonic()
        while True:
            resp = self._client.get_results(project_id, worker_id)
            if resp.status_code == 200:
                return resp.json()
            logger.debug("Waiting for global results...")
            time.sleep(interval)
            if timeout is not None and time.monotonic() - start >= timeout:
                msg = f"Timed out waiting for results for project {project_id}"
                raise TimeoutError(msg)
