"""DC-Cox regression use case."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dccox.block import BlockMatrix
from dccox.cox import Projector, Regressor, SurvivalFunction
from dccox.utils import validate_methods


@validate_methods
class Horizontal:
    """DC-Cox regression use case."""

    def global_create_Xanc(self, n_features: int, *, r: int = 100) -> np.ndarray:
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
        Xanc : np.array
            The anchor matrix with shape (r, n_features).
        """
        return np.random.randn(r, n_features)

    def local_load_metadata(
        self,
        clinical_data_path: str,
        *,
        keep_feature_cols: list[str] | None = None,
        meta_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
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
        X : np.array
            The feature matrix with shape (n_samples, n_features).
        y : np.array
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

        print(metadata)

        # Deal with the target columns
        expected_cols = ["time", "event"]
        missed_cols = []
        for col in expected_cols:
            if col not in metadata.columns:
                missed_cols.append(col)
        assert len(missed_cols) == 0, f"Missing columns {missed_cols}."

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

        # Deal with the sample metadata columns
        meta = metadata.loc[:, meta_cols]

        # Remove samples with the missing values
        metadata = metadata.loc[:, expected_cols + keep_feature_cols]
        metadata = metadata.dropna()
        if len(metadata) == 0:
            logging.warning("There are no samples left.")

        print(
            f"\033[95m The feature matrix X:\n \033[0m"
            f"{metadata.loc[:, keep_feature_cols]}\n"
            f"\n"
            f"\033[95m The target matrix y:\n \033[0m"
            f"{metadata.loc[:, expected_cols]}\n"
            f"\n"
            f"\033[95m The metadata:\n \033[0m"
            f"{meta}\n"
        )

        X = metadata.loc[:, keep_feature_cols].to_numpy()
        y = metadata.loc[:, expected_cols].to_numpy()
        return X, y, keep_feature_cols, meta

    def local_create_proxy_data(
        self,
        X: np.ndarray,
        Xanc: np.ndarray,
        y: np.ndarray,
        k: int = 20,
        bs_prop: float = 0.6,
        bs_times: int = 20,
        bs_replace: bool = False,
        alpha: float = 0.05,
        step_size: float = 0.5,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Create linear projection matrix F, projected matrix X_tilde and projected anchor matrix Xanc_tilde.

        Parameters
        ----------
        X : np.array
            The feature matrix with shape (n_samples, n_features).
        Xanc : np.array
            The global anchor matrix with shape (r, n_features).
            where r is the pseudo-number of samples of Xanc.
        y : np.array
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
        F : np.array
            The linear projection matrix to be used for creating the projected matrices,
            X_tilde and Xanc_tilde, its shape is (n_features, m tilde).
        X_tilde : list of np.array
            The projected feature matrix with shape (n_samples, m tilde).
        Xanc_tilde : list of np.array
            The projected global anchor matrix with shape (r, m tilde).
        feature_sum : np.array
            The sums of the features.
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
        coef: np.ndarray, coef_var: np.ndarray, Gs: BlockMatrix
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
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
        Xs_tilde: list[list[np.ndarray | None]],
        Xancs_tilde: list[list[np.ndarray | None]],
        ys: list[np.ndarray],
        sums: list[np.ndarray],
        alpha: float = 0.05,
        step_size: float = 0.5,
        var_thres: float = 1e-8,
        m_hat: int | None = None,
    ) -> tuple[
        list[list[np.ndarray]], list[list[np.ndarray]], pd.DataFrame, np.ndarray
    ]:
        """
        Perform the Cox-PH regression on the given Xs_tilde and Xancs_tilde.

        Parameters
        ----------
        Xs_tilde : list of lists of np.array
            The data structure is:
            [
                [X_tilde],
                [X_tilde],...
            ]
            The shape for each projected feature matrix is (nc, m tilde).
        Xancs_tilde : list of lists of np.array | list of lists of None
            The data structure is:
            [
                [Xanc_tilde],
                [Xanc_tilde],...
            ]
            The shape for each projected global anchor matrix is (r, m tilde).
        ys : list of np.array
            The concatenated time and event vectors with shape (nc, 2).
        sums : list of np.array
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
        coef : list of list of np.array
            The beta tilde vector with shape (m tilde,).
        coef_var : list of list of np.array
            The variance-covariance matrix with shape (m tilde, m tilde).
        baseline_hazard : pd.DataFrame
            The baseline hazard values, where the index is the time.
        feature_mean : np.array
            The global means of the features.
        """
        assert isinstance(Xs_tilde[0], list), "Xs_tilde must be a list of lists."
        assert isinstance(Xancs_tilde[0], list), "Xancs_tilde must be a list of lists."
        assert isinstance(ys, list), "y must be a list of 2d arrays."
        assert isinstance(sums, list), "sums must be a list of 1d arrays."
        keep_idx = [i for i, Xc in enumerate(Xs_tilde) if Xc[0] is not None]

        Xs_tilde = BlockMatrix([Xs_tilde[idx] for idx in keep_idx])
        Xancs_tilde = BlockMatrix([Xancs_tilde[idx] for idx in keep_idx])
        ys = np.concatenate([ys[idx] for idx in keep_idx])
        feature_mean = np.sum(sums, axis=0) / ys.shape[0]

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
        coef: np.ndarray,
        coef_var: np.ndarray,
        baseline_hazard: pd.DataFrame,
        mean: np.ndarray,
        F: np.ndarray,
        alpha: float = 0.05,
    ) -> SurvivalFunction:
        """
        Recover the survival function and statistical properties from projected coefficients and variance-covariance matrix.

        Parameters
        ----------
        keep_feature_cols : list of strings
            The features to be perform Cox-PH regression.
        coef : np.array
            The beta tilde vector with shape (m tilde,).
        coef_var : np.array
            The variance-covariance matrix with shape (m tilde, m tilde).
        baseline_hazard : pd.DataFrame
            The baseline hazard values, where the index is the time.
        mean : np.array
            The global means of the features.
        F : np.array
            The linear projection matrix to be used for creating the projected matrices,
            X_tilde and Xanc_tilde, its shape is (n_features, m tilde).

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
            coef, coef_var, baseline_hazard, mean, alpha=alpha
        )
        return survival_func
