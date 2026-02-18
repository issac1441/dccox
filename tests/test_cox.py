"""Test distributed federated Cox regression with various BlockMatrix configurations."""

from __future__ import annotations

import unittest

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import numpy as np
import pandas as pd

from dccox.block import BlockMatrix
from dccox.cox import Projector, Regressor


def _build_ground_truth(
    X: np.ndarray, durations: np.ndarray, events: np.ndarray
) -> CoxPHFitter:
    """Fit a centralized CoxPHFitter as ground truth."""
    data = pd.concat(
        [
            pd.DataFrame(X),
            pd.DataFrame({"duration": durations, "event": events}),
        ],
        axis=1,
    )
    model = CoxPHFitter(alpha=0.05)
    model.fit(
        data, duration_col="duration", event_col="event", fit_options={"step_size": 0.5}
    )
    return model


def _run_distributed(
    X: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    sample_splits: list[int],
    feature_splits: list[int],
    r: int = 100,
) -> tuple[Regressor, list[list[np.ndarray]]]:
    """Run the distributed DC-Cox pipeline for given sample/feature splits.

    Returns the fitted Regressor and the list of projection matrices Fs[c][d].
    """
    n_clients = len(sample_splits)
    n_feat_blocks = len(feature_splits)

    # Split samples
    sample_idxs = np.cumsum([0, *sample_splits])
    # Split features
    feat_idxs = np.cumsum([0, *feature_splits])

    # Generate one global Xanc per feature block
    Xancs = [np.random.randn(r, fd) for fd in feature_splits]

    Xs_tilde_blocks: list[list[np.ndarray]] = []
    Xancs_tilde_blocks: list[list[np.ndarray]] = []
    Fs: list[list[np.ndarray]] = []

    for c in range(n_clients):
        Xc = X[sample_idxs[c] : sample_idxs[c + 1], :]
        dur_c = durations[sample_idxs[c] : sample_idxs[c + 1]]
        evt_c = events[sample_idxs[c] : sample_idxs[c + 1]]

        Xc_tilde_row = []
        Xanc_tilde_row = []
        Fc_row = []
        for d in range(n_feat_blocks):
            Xcd = Xc[:, feat_idxs[d] : feat_idxs[d + 1]]
            Xanc_d = Xancs[d]

            projector = Projector(
                k=20, bs_prop=0.6, bs_times=20, alpha=0.05, step_size=0.5
            )
            projector.project(X=Xcd, Xanc=Xanc_d, events=evt_c, durations=dur_c)

            Xc_tilde_row.append(projector.X_tilde)
            Xanc_tilde_row.append(projector.Xanc_tilde)
            Fc_row.append(projector.F)

        Xs_tilde_blocks.append(Xc_tilde_row)
        Xancs_tilde_blocks.append(Xanc_tilde_row)
        Fs.append(Fc_row)

    Xs_tilde = BlockMatrix(Xs_tilde_blocks)
    Xancs_tilde = BlockMatrix(Xancs_tilde_blocks)

    all_durations = durations
    all_events = events

    model = Regressor(alpha=0.05, step_size=0.5).fit(
        Xs_tilde=Xs_tilde,
        Xancs_tilde=Xancs_tilde,
        durations=all_durations,
        events=all_events,
    )

    return model, Fs


def _recover_coef(
    model: Regressor, Fs: list[list[np.ndarray]], feature_splits: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Recover original-space coefficients from the fitted distributed model.

    Uses client 0 for recovery (any client should give the same result).
    """
    Gs = model.Gs
    coef_hat = model.coef
    coef_var_hat = model.coef_var

    # Recover coef_tilde per block for client 0
    coef_tilde_parts = []
    coef_var_tilde_parts = []
    for d in range(Gs.shape[1]):
        G_0d = Gs[0, d]
        coef_tilde_parts.append(G_0d @ coef_hat)
        coef_var_tilde_parts.append(G_0d @ coef_var_hat @ G_0d.T)

    # Recover original space via F: coef_original_d = F_0d @ coef_tilde_d
    coef_parts = []
    coef_var_parts = []
    for d in range(len(feature_splits)):
        F_0d = Fs[0][d]
        coef_parts.append(F_0d @ coef_tilde_parts[d])
        coef_var_parts.append(F_0d @ coef_var_tilde_parts[d] @ F_0d.T)

    coef_recovered = np.concatenate(coef_parts)
    # Build block-diagonal var-cov matrix
    coef_var_recovered = np.zeros((sum(feature_splits), sum(feature_splits)))
    idx = 0
    for d, fs in enumerate(feature_splits):
        coef_var_recovered[idx : idx + fs, idx : idx + fs] = coef_var_parts[d]
        idx += fs

    return coef_recovered, coef_var_recovered


class DistributedRegressorTestCase(unittest.TestCase):
    """Test Regressor with various c x d BlockMatrix configurations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up shared test data from the Rossi dataset."""
        np.random.seed(42)
        data = load_rossi()
        data.columns = ["time", "event", *data.columns[2:]]

        # Use first 360 samples, all 7 features
        cls.n_samples = 360
        cls.n_features = len(data.columns) - 2
        cls.X = data.iloc[: cls.n_samples, 2:].to_numpy().astype(float)
        cls.durations = data.iloc[: cls.n_samples, 0].to_numpy().astype(float)
        cls.events = data.iloc[: cls.n_samples, 1].to_numpy().astype(float)
        cls.feature_names = list(data.columns[2:])

        # Ground truth
        cls.gt = _build_ground_truth(cls.X, cls.durations, cls.events)

    def _assert_coef_close(
        self, sample_splits: list[int], feature_splits: list[int]
    ) -> None:
        """Run distributed pipeline and assert recovered coef matches ground truth."""
        # Make the randomized parts of the pipeline deterministic per test case.
        # (Anchors + bootstrap sampling rely on numpy's global RNG.)
        np.random.seed(42)
        model, Fs = _run_distributed(
            self.X, self.durations, self.events, sample_splits, feature_splits
        )
        coef_recovered, coef_var_recovered = _recover_coef(model, Fs, feature_splits)

        gt_coef = self.gt.summary["coef"].to_numpy()
        gt_var = np.diag(self.gt.variance_matrix_)

        # Numerical optimization + SVD/projection can cause tiny floating-point differences.
        np.testing.assert_array_almost_equal(coef_recovered, gt_coef, decimal=4)
        np.testing.assert_array_almost_equal(
            np.diag(coef_var_recovered), gt_var, decimal=4
        )

        # Baseline hazard should match up to small numerical noise
        pd.testing.assert_frame_equal(
            model.baseline_hazard,
            self.gt.baseline_hazard_,
            check_exact=False,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_regressor_1x1(self) -> None:
        """Test 1 client, 1 feature block (baseline)."""
        self._assert_coef_close(
            sample_splits=[360],
            feature_splits=[7],
        )

    def test_regressor_3x1(self) -> None:
        """Test 3 clients, 1 feature block."""
        self._assert_coef_close(
            sample_splits=[100, 120, 140],
            feature_splits=[7],
        )

    def test_regressor_1x3(self) -> None:
        """Test 1 client, 3 feature blocks."""
        self._assert_coef_close(
            sample_splits=[360],
            feature_splits=[3, 2, 2],
        )

    def test_regressor_3x3(self) -> None:
        """Test 3 clients, 3 feature blocks (full distributed)."""
        self._assert_coef_close(
            sample_splits=[100, 120, 140],
            feature_splits=[3, 2, 2],
        )

    def test_regressor_2x4(self) -> None:
        """Test 2 clients, 4 feature blocks."""
        self._assert_coef_close(
            sample_splits=[180, 180],
            feature_splits=[2, 2, 2, 1],
        )

    def test_regressor_4x2(self) -> None:
        """Test 4 clients, 2 feature blocks."""
        self._assert_coef_close(
            sample_splits=[80, 100, 90, 90],
            feature_splits=[4, 3],
        )
