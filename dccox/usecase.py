"""DC-Cox regression use case."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from dccox.block import BlockMatrix
from dccox.cox import Projector, Regressor, SurvivalFunction
from dccox.types import Array1D, Array2D
from dccox.utils import validate_methods


@validate_methods
class Horizontal:
    """DC-Cox regression use case."""

    def global_create_Xanc(self, n_features: int, *, r: int = 100) -> Array2D:
        """
        Generate a random anchor matrix Xanc.

        Parameters
        ----------
        n_features : int
            The number of features to be perform Cox-PH regression.
        r : int
            The pseudo-number of samples of Xanc.

        Returns
        -------
        Xanc : Array2D
            The anchor matrix with shape (r, n_features).
        """
        return np.random.randn(r, n_features)

    def local_load_metadata(
        self,
        clinical_data_path: str,
        *,
        keep_feature_cols: list[str] | None = None,
        meta_cols: list[str] | None = None,
    ) -> tuple[Array2D, Array2D, list[str], pd.DataFrame]:
        """
        Load clinical metadata.

        The sample will be dropped if it has any missing values.

        Parameters
        ----------
        clinical_data_path : str
            The path to clinical metadata.
        keep_feature_cols : list of strings
            The features to be perform Cox-PH regression.
        meta_cols : list of str
            The columns recording the sample information.

        Returns
        -------
        X : Array2D
            The feature matrix with shape (n_samples, n_features).
        y : Array2D
            The concatenated time and event vectors with shape (n_samples, 2).
        keep_feature_cols : list of str
            The features to be perform Cox-PH regression.
        meta: pd.DataFrame
            The sample metadata.
        """
        metadata = pd.read_csv(clinical_data_path)

        if "Unnamed: 0" in metadata.columns:
            metadata.set_index("Unnamed: 0", inplace=True)
            metadata.index.name = None

        logging.debug(f"Loaded metadata:\n{metadata}")

        # Deal with the target columns
        expected_cols = ["time", "event"]
        missed_cols = []
        for col in expected_cols:
            if col not in metadata.columns:
                missed_cols.append(col)
        if len(missed_cols) > 0:
            raise ValueError(f"Missing columns {missed_cols}.")

        meta_cols = [] if meta_cols is None else meta_cols

        # Deal with the feature columns
        if keep_feature_cols is None:
            keep_feature_cols = list(
                filter(lambda x: x not in expected_cols + meta_cols, metadata.columns)
            )
        else:
            unexpected_features = set(keep_feature_cols).difference(metadata.columns)
            if len(unexpected_features) > 0:
                logging.warning(
                    f"The feature columns {unexpected_features} are not in the clinical data. "
                    f"They are removed automatically."
                )
            keep_feature_cols = list(
                filter(lambda x: x in metadata.columns, keep_feature_cols)
            )

        # Remove samples with the missing values
        metadata = metadata.loc[:, expected_cols + keep_feature_cols + meta_cols]
        metadata = metadata.dropna()

        # Deal with the sample metadata columns (after dropna to preserve index alignment)
        meta = metadata.loc[:, meta_cols]
        if len(metadata) == 0:
            logging.warning("There are no samples left.")

        logging.info(
            f"Feature matrix X shape: {metadata.loc[:, keep_feature_cols].shape}, "
            f"Target matrix y shape: {metadata.loc[:, expected_cols].shape}, "
            f"Metadata shape: {meta.shape}"
        )

        X = metadata.loc[:, keep_feature_cols].to_numpy()
        y = metadata.loc[:, expected_cols].to_numpy()
        return X, y, keep_feature_cols, meta

    def local_create_proxy_data(
        self,
        X: Array2D,
        Xanc: Array2D,
        y: Array2D,
        *,
        k: int = 20,
        bs_prop: float = 0.6,
        bs_times: int = 20,
        bs_replace: bool = False,
        alpha: float = 0.05,
        step_size: float = 0.5,
    ) -> (
        tuple[Array2D, list[Array2D], list[Array2D], Array1D]
        | tuple[None, list[None], list[None], Array1D]
    ):
        """
        Create linear projection matrix F, projected matrix X_tilde and projected anchor matrix Xanc_tilde.

        Parameters
        ----------
        X : Array2D
            The feature matrix with shape (n_samples, n_features).
        Xanc : Array2D
            The global anchor matrix with shape (r, n_features).
            where r is the pseudo-number of samples of Xanc.
        y : Array2D
            The concatenated time and event vectors with shape (n_samples, 2).
        k : int
            The latent dimension of Fdr. Notice that the ultimate dimension is min(k, len(S)),
            where the S is the number of nonzero singular values.
        bs_prop : float
            The proportion of the samples in a client used for bootstrapping for each time.
        bs_times : int
            The number of times to bootstrap.
        bs_replace : bool
            Whether to sample with replacement when generating bootstrap subsets for Fbs.
            - True: classic bootstrap (with replacement).
            - False: subsampling (without replacement).
        alpha : float
            The level in the confidence intervals.
            It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
        step_size : float
            Deal with the fitting error, `delta contains nan value(s)`.
            See also: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model

        Returns
        -------
        F : Array2D
            The linear projection matrix to be used for creating the projected matrices,
            X_tilde and Xanc_tilde, its shape is (n_features, m tilde).
        X_tilde : list of Array2D
            The projected feature matrix with shape (n_samples, m tilde).
        Xanc_tilde : list of Array2D
            The projected global anchor matrix with shape (r, m tilde).
        feature_sum : Array1D
            The sums of the features.

        Notes
        -----
        In Algorithm 1, each worker also shares (t_i, delta_i) to the master along with (X_tilde, Xanc_tilde).
        In this codebase, y (concatenated t and delta) is returned separately as ys in the orchestration layer.
        """
        if len(X) == 0:
            return None, [None], [None], np.zeros(X.shape[1])

        projector = Projector(
            k=k,
            bs_prop=bs_prop,
            bs_times=bs_times,
            bs_replace=bs_replace,
            alpha=alpha,
            step_size=step_size,
        )
        projector.project(X=X, Xanc=Xanc, durations=y[:, 0], events=y[:, 1])

        F = projector.F
        X_tilde = projector.X_tilde
        Xanc_tilde = projector.Xanc_tilde

        feature_sum = np.sum(X, axis=0)

        return F, [X_tilde], [Xanc_tilde], feature_sum

    @staticmethod
    def _global_compute_coef_tilde(
        coef: Array1D, coef_var: Array2D, Gs: BlockMatrix
    ) -> tuple[list[list[Array1D]], list[list[Array2D]]]:
        coef_ = [
            [Gs[c, d] @ coef for d in range(Gs.shape[1])] for c in range(Gs.shape[0])
        ]

        coef_var_ = [
            [Gs[c, d] @ coef_var @ Gs[c, d].T for d in range(Gs.shape[1])]
            for c in range(Gs.shape[0])
        ]
        return coef_, coef_var_

    def global_fit_model(
        self,
        Xs_tilde: list[list[Array2D | None]],
        Xancs_tilde: list[list[Array2D | None]],
        ys: list[Array2D],
        sums: list[Array1D],
        *,
        alpha: float = 0.05,
        step_size: float = 0.5,
        var_thres: float = 1e-8,
        m_hat: int | None = None,
    ) -> tuple[list[list[Array1D]], list[list[Array2D]], pd.DataFrame, Array1D]:
        """
        Perform the Cox-PH regression on the given Xs_tilde and Xancs_tilde.

        Parameters
        ----------
        Xs_tilde : list of lists of Array2D
            The data structure is:
            [
                [X_tilde],
                [X_tilde],...
            ]
            The shape for each projected feature matrix is (nc, m tilde).
        Xancs_tilde : list of lists of Array2D | list of lists of None
            The data structure is:
            [
                [Xanc_tilde],
                [Xanc_tilde],...
            ]
            The shape for each projected global anchor matrix is (r, m tilde).
        ys : list of Array2D
            The concatenated time and event vectors with shape (nc, 2).
        sums : list of Array1D
            The sums of the features.
        alpha : float
            The level in the confidence intervals.
            It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
        step_size : float
            Deal with the fitting error, `delta contains nan value(s)`.
            See also: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        var_thres : float
            The threshold for dropping low variance columns.
        m_hat : int | None
            The hyperparameter controlling the dimension of the transformed feature space.
            If None, the dimension is determined by the number of non-zero singular values.

        Returns
        -------
        coef : list of list of Array1D
            The beta tilde vector with shape (m tilde,).
        coef_var : list of list of Array2D
            The variance-covariance matrix with shape (m tilde, m tilde).
        baseline_hazard : pd.DataFrame
            The baseline hazard values, where the index is the time.
        feature_mean : Array1D
            The global means of the features.
            Note: This is an implementation convenience for prediction/reporting (e.g., to align with lifelines baseline hazard convention),
            and is not a core artifact of the DC-Cox paper.
        """
        if not isinstance(Xs_tilde[0], list):
            raise TypeError("Xs_tilde must be a list of lists.")
        if not isinstance(Xancs_tilde[0], list):
            raise TypeError("Xancs_tilde must be a list of lists.")
        if not isinstance(ys, list):
            raise TypeError("ys must be a list of 2d arrays.")
        if not isinstance(sums, list):
            raise TypeError("sums must be a list of 1d arrays.")
        keep_idx = [i for i, Xc in enumerate(Xs_tilde) if Xc[0] is not None]

        Xs_tilde = BlockMatrix([Xs_tilde[idx] for idx in keep_idx])
        Xancs_tilde = BlockMatrix([Xancs_tilde[idx] for idx in keep_idx])
        ys = np.concatenate([ys[idx] for idx in keep_idx])

        # Keep numerator/denominator consistent with filtered clients.
        kept_sums = [sums[idx] for idx in keep_idx]
        feature_mean = np.sum(kept_sums, axis=0) / ys.shape[0]

        model = Regressor(
            alpha=alpha,
            step_size=step_size,
            var_thres=var_thres,
            m_hat=m_hat,
        ).fit(
            Xs_tilde=Xs_tilde,
            Xancs_tilde=Xancs_tilde,
            events=ys[:, 1],
            durations=ys[:, 0],
        )

        coef, coef_var = self._global_compute_coef_tilde(
            model.coef, model.coef_var, model.Gs
        )
        baseline_hazard = model.baseline_hazard

        return coef, coef_var, baseline_hazard, feature_mean

    def local_recover_survival(
        self,
        keep_feature_cols: list[str],
        coef: Array1D,
        coef_var: Array2D,
        baseline_hazard: pd.DataFrame,
        mean: Array1D,
        F: Array2D,
        *,
        alpha: float = 0.05,
        centering: Literal["mean"] | None = None,
    ) -> SurvivalFunction:
        """
        Recover the survival function and statistical properties from projected coefficients and variance-covariance matrix.

        Parameters
        ----------
        keep_feature_cols : list of strings
            The features to be perform Cox-PH regression.
        coef : Array1D
            The beta tilde vector with shape (m tilde,).
        coef_var : Array2D
            The variance-covariance matrix with shape (m tilde, m tilde).
        baseline_hazard : pd.DataFrame
            The baseline hazard values, where the index is the time.
        mean : Array1D
            The global means of the features.
            Note: This is an implementation convenience for prediction/reporting (e.g., to align with lifelines baseline hazard convention),
            and is not a core artifact of the DC-Cox paper.
        F : Array2D
            The linear projection matrix to be used for creating the projected matrices,
            X_tilde and Xanc_tilde, its shape is (n_features, m tilde).
        alpha : float
            The significance level for confidence intervals.
        centering : Literal["mean"] | None
            The centering strategy for baseline hazard computation.
            - None (default): baseline hazard at x=0 (paper convention)
            - "mean": baseline hazard at x=mean (lifelines convention)
            See SurvivalFunction docstring for detailed explanation.

        Returns
        -------
        survival_func : SurvivalFunction
            The object recovers the survival function and statistical properties
            from the given coefficients and variance-covariance matrix.
        """
        coef = F @ coef
        coef_var = F @ coef_var @ F.T
        coef = pd.Series(coef, index=keep_feature_cols)
        coef.name = "covariate"
        survival_func = SurvivalFunction(
            coef, coef_var, baseline_hazard, mean, alpha=alpha, centering=centering
        )
        return survival_func
