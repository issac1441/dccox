"""DC-Cox PH Regression."""

from __future__ import annotations

from typing import Literal, Self

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import scipy as sp

try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

from dccox.block import BlockMatrix
from dccox.types import Array1D, Array2D
from dccox.utils import validate_methods


@validate_methods
class Projector:
    """
    Project data for DC-Cox PH Regression.

    Attributes
    ----------
    k : int
        The latent dimension of Fdr. Notice that the ultimate dimension is min(k, len(S)),
        where the S is the number of nonzero singular values.
    bs_prop : float
        The proportion of the samples in a client used for bootstrapping for each time.
    bs_times : int
        The number of times to bootstrap.
    fitter_params : dict
        Parameters for the CoxPHFitter.
    fitting_params : dict
        Parameters for the fitting process.
    """

    def __init__(
        self,
        *,
        k: int = 10,
        bs_prop: float = 0.6,
        bs_times: int = 20,
        bs_replace: bool = False,
        alpha: float = 0.05,
        step_size: float = 0.5,
    ) -> None:
        """
        Project data for DC-Cox PH Regression.

        Parameters
        ----------
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

        Notes
        -----
        Setting `bs_replace=False` is generally more stable for Cox fitting. Sampling with replacement can
        produce many duplicate rows and an effectively smaller sample size, which increases the chance of
        high collinearity / separation and may trigger a singular-matrix convergence error in lifelines.

        See Also
        --------
        `lifelines.fitters.coxph_fitter.CoxPHFitter`
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        DC-COX[https://doi.org/10.1016/j.jbi.2022.104264]
        """
        self.k = k
        self.bs_prop = bs_prop
        self.bs_times = bs_times
        self.bs_replace = bs_replace
        self.fitter_params = {"alpha": alpha}
        self.fitting_params = {
            "duration_col": "duration",
            "event_col": "event",
            "fit_options": {"step_size": step_size},
        }

    @property
    def F(self) -> Array2D:
        """
        Get the transformation matrix F.

        Returns
        -------
        Array2D
            The transformation matrix F with shape (n_features, m_tilde).
        """
        return self.__F

    @property
    def X_tilde(self) -> Array2D:
        """
        Get the transformed feature matrix X tilde.

        Returns
        -------
        Array2D
            The transformed feature matrix X tilde with shape (n_samples, m_tilde).
        """
        return self.__X_tilde

    @property
    def Xanc_tilde(self) -> Array2D:
        """
        Get the transformed global anchor matrix Xanc tilde.

        Returns
        -------
        Array2D
            The transformed global anchor matrix Xanc tilde with shape (r, m_tilde).
        """
        return self.__Xanc_tilde

    def _compute_Fbs(self, X: Array2D, events: Array1D, durations: Array1D) -> Array2D:
        """
        Apply the Cox PH model to X, events and durations to obtain coefficients.

        Parameters
        ----------
        X : Array2D
            The feature matrix with shape (n_samples, n_features).
        events : Array1D
            The one-hot event vector with shape (n_samples,).
        durations : Array1D
            The one-hot duration vector with shape (n_samples,).

        Returns
        -------
        Fbs : Array2D
            The concatenated betas from each bootstrap regression with shape (n_features, n_times).
        """
        coefs = []
        n_samples = X.shape[0]
        X = pd.DataFrame(X)
        y = pd.DataFrame({"duration": durations, "event": events})
        data = pd.concat([X, y], axis=1)
        for _ in range(self.bs_times):
            idx = np.random.choice(
                n_samples, int(self.bs_prop * n_samples), replace=self.bs_replace
            )
            model = CoxPHFitter(**self.fitter_params)
            model.fit(data.iloc[idx, :], **self.fitting_params)
            coef_ = model.summary["coef"].to_numpy()
            coefs.append(coef_.reshape(1, -1))
        Fbs = np.concatenate(coefs).T
        return Fbs

    def _compute_Fdr(self, X: Array2D) -> Array2D:
        """
        Compute a feature-space dimensionality reduction basis F_DR (e.g., PCA loadings).

        F_DR is in R^{m x m_tilde_DR}.
        It maps the input features to the reduced feature space.

        Parameters
        ----------
        X : Array2D
            The feature matrix with shape (n_samples, n_features).

        Returns
        -------
        Fdr : Array2D
            The PCs with shape (n_features, min(k, len(non-zero singular values))).
        """
        X_shift = (X - X.mean(axis=0, keepdims=True)).T
        U, S, _ = np.linalg.svd(X_shift, full_matrices=False)
        k = min(self.k, len(S))
        return U[:, :k] * S[:k]

    @staticmethod
    def _create_random_nonsingular(n: int) -> Array2D:
        """
        Create a random nonsingular matrix.

        The paper uses a nonsingular random matrix E. Here we use an orthogonal matrix (QR decomposition)
        which is a numerically stable special case of a nonsingular matrix.
        """
        # Random orthogonal matrix via QR factorization
        A = np.random.randn(n, n)
        Q, R = np.linalg.qr(A)
        # Fix sign ambiguity so it's a proper random draw over O(n)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs
        return Q

    def _compute_F(self, X: Array2D, events: Array1D, durations: Array1D) -> None:
        # n_features x mbs(bs_times)
        Fbs = self._compute_Fbs(X, events, durations)
        # n_features x mdr(min(k, len(non-zero singular values)))
        Fdr = self._compute_Fdr(X)
        # n_features x m tilde(mbs+mdr)
        F_ = np.concatenate([Fbs, Fdr], axis=1)
        # m tilde x m tilde
        E = self._create_random_nonsingular(F_.shape[1])
        # n_features x m tilde
        self.__F = F_ @ E

    def project(
        self, X: Array2D, Xanc: Array2D, events: Array1D, durations: Array1D
    ) -> None:
        """
        Perform the linear transformation.

        X_tilde = X @ F, where for a worker (i,j), X_{i,j} is in R^{n_i x m_j}.
        We compute X_tilde_{i,j} = X_{i,j} F_{i,j} and Xanc_tilde_{i,j} = Xanc_{:,j} F_{i,j}.
        Xanc tilde = Xanc @ F, where Xanc shape is (r, md).
        F = [Fbs, Fdr], where Fbs shape is (md, bs_times)
        and Fdr shape is (md, min(k, len(non-zero singular values))).

        Parameters
        ----------
        X : Array2D
            The feature matrix with shape (nc, md),
            where nc is the number of samples in Xc,: and md is the number of features in X:,d.
        Xanc : Array2D
            The global anchor matrix with shape (r, md).
            where r is the pseudo-number of samples of Xanc and md is the number of features in X:,d.
        events : Array1D
            The one-hot event vecotr with shape (nc,).
        durations : Array1D
            The time vector with shape (nc,).
        """
        self._compute_F(X, events, durations)
        # n_samples x m tilde
        self.__X_tilde = X @ self.__F
        # r x m tilde
        self.__Xanc_tilde = Xanc @ self.__F


@validate_methods
class Regressor:
    """Perform DC-Cox PH Regression."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        step_size: float = 0.5,
        var_thres: float = 1e-8,
        m_hat: int | None = None,
    ) -> None:
        """
        Perform DC-Cox PH Regression.

        Parameters
        ----------
        alpha : float
            The level in the confidence intervals.
            It is `alpha` parameter in `lifelines.fitters.coxph_fitter.CoxPHFitter`.
        step_size : float
            Deal with the fitting error, `delta contains nan value(s)`.
        var_thres : float
            The threshold for dropping low variance columns.
        m_hat : int | None
            The hyperparameter controlling the dimension of the transformed feature space.
            If None, the dimension is determined by the number of non-zero singular values.

        See Also
        --------
        `lifelines.fitters.coxph_fitter.CoxPHFitter`
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        DC-COX[https://doi.org/10.1016/j.jbi.2022.104264]
        """
        self.var_thres = var_thres
        self.m_hat = m_hat
        self.fitter_params = {"alpha": alpha}
        self.fitting_params = {
            "duration_col": "duration",
            "event_col": "event",
            "fit_options": {"step_size": step_size},
        }

    @property
    def coef(self) -> Array1D:
        r"""
        Coefficients.

        .. math::
            \beta
        """
        return self.__coef

    @property
    def coef_var(self) -> Array2D:
        r"""
        Variance-covariance matrix.

        .. math::
            \mathrm{var}\left(\beta\right)
        """
        return self.__coef_var

    @property
    def baseline_hazard(self) -> pd.DataFrame:
        r"""
        Baseline hazards.

        .. math::
            h_0\\left(t\right)
        """
        return self.__baseline_hazard

    @property
    def Gs(self) -> list[Array2D]:
        r"""
        Target matrices.

        .. math::
            G_c
        """
        return self.__Gs

    def _compute_target_matrix(self, Xancs_tilde: BlockMatrix) -> BlockMatrix:
        """
        Compute the target matrix Gs.

        [Xancs_tilde_1, ..., Xancs_tilde_c] = U * S @ Vh.
        Gc = pinv(Xanc_tilde_c) @ P

        Let [Xanc_tilde_1, ..., Xanc_tilde_c] approx U_hat_m S_hat_m V_hat_m^T.
        We form P = U_hat_m S_hat_m (corresponding to choosing C = S_hat_m in Eq.(5)),
        and compute G_i = (Xanc_tilde_i)^dagger P.

        Parameters
        ----------
        Xancs_tilde : BlockMatrix
            The `BlockMatrix` with shape (c, d).

        Returns
        -------
        Gs : BlockMatrix
            The `BlockMatrix(axis=0)`.
            For each G in BlockMatrix, the shape is (mc tilde, m hat).
        """
        # r x (c * m tilde), where m tilde equals to the sum of Fbs.shape[1]+Fdr.shape[1] for all c.
        # e.g. c1d1 = 20+4, c1d2 = 20+3, m tilde = 47
        Xancs_ = np.concatenate(
            [Xancs_tilde[c, :] for c in range(Xancs_tilde.shape[0])], axis=1
        )

        # U: (r x m hat), m hat equals to len(S)
        U, S, _ = np.linalg.svd(Xancs_)

        m_hat = self.m_hat if self.m_hat is not None else len(S)
        P = U[:, :m_hat] * S[:m_hat]

        Gs = []
        for c in range(Xancs_tilde.shape[0]):
            # Gc: m tilde x m hat
            Gc = np.linalg.pinv(Xancs_tilde[c, :]) @ P
            # Use client c's dsizes, not blocks[0]'s
            dsizes_c = [
                Xancs_tilde.blocks[c][d].shape[1] for d in range(Xancs_tilde.shape[1])
            ]
            idxs = np.cumsum([0, *dsizes_c])
            # Gcd: mc tilde x m hat
            Gs.append([Gc[idxs[i] : idxs[i + 1], :] for i in range(len(idxs) - 1)])

        return BlockMatrix(Gs, axis=0)

    def _drop_low_var_cols(
        self, X_hat: Array2D, Gs: BlockMatrix
    ) -> tuple[Array2D, BlockMatrix]:
        vars_ = np.var(X_hat, axis=0)
        keeps_ = vars_ > self.var_thres
        X_hat = X_hat[:, keeps_]
        for c in range(Gs.shape[0]):
            for d in range(Gs.shape[1]):
                Gs.blocks[c][d] = Gs.blocks[c][d][:, keeps_]
        return X_hat, Gs

    def _cox_regression(
        self,
        Xs_tilde: BlockMatrix,
        Gs: BlockMatrix,
        durations: Array1D,
        events: Array1D,
    ) -> None:
        """
        Perform Cox-PH regression on X hat.

        X hat = concat([Xc_tilde @ Gc]).
        CoxPHFitter().fit(X hat, y).

        Parameters
        ----------
        Xs_tilde : BlockMatrix
            The proxy data with shape (c, d).
        Gs : BlockMatrix
            The `BlockMatrix(axis=0)`.
            For each G in BlockMatrix, the shape is (mc tilde, m hat).
        durations : Array1D
            The duration time vector with shape (n,).
        events : Array1D
            The one-hot event vector with shape (n,).
            e.g. 1 for dead, 0 for survived.
        """
        X_hat = np.concatenate(
            [Xs_tilde[c, :] @ Gs[c, :] for c in range(Xs_tilde.shape[0])]
        )
        X_hat, Gs = self._drop_low_var_cols(X_hat, Gs)
        X_hat = pd.DataFrame(X_hat)
        y = pd.DataFrame({"duration": durations, "event": events})
        model = CoxPHFitter(**self.fitter_params)
        model.fit(pd.concat([X_hat, y], axis=1), **self.fitting_params)
        self.__baseline_hazard = model.baseline_hazard_
        self.__coef = model.summary["coef"].to_numpy()
        self.__coef_var = model.variance_matrix_.to_numpy()
        self.__Gs = Gs

    def fit(
        self,
        Xs_tilde: BlockMatrix,
        Xancs_tilde: BlockMatrix,
        durations: Array1D,
        events: Array1D,
    ) -> Self:
        r"""
        Calculate the target matrix G and perform the Cox-PH regression.

        .. math::
            \begin{align*}
            &\\left[\tilde{X}_1^{\\mathrm{anc}}, \tilde{X}_2^{\\mathrm{anc}},...,\tilde{X}_c^{\\mathrm{anc}}\right] \\
            & \approx U_{\\hat{m}} \\Sigma_{\\hat{m}}  V_{\\hat{m}}^{T}
            G_i = (\tilde{X}^{\\mathrm{anc}}_i)^\\dagger U_{\\hat{m}}C
            \begin{equation}
            \\hat{X} = \begin{bmatrix}
            \\hat{X}_1 \\
            \\hat{X}_2 \\
            \vdots \\
            \\hat{X}_n
            \\end{bmatrix} = \begin{bmatrix}
            \tilde{X}_1 \\mathbf{G}_1 \\
            \tilde{X}_2 \\mathbf{G}_2 \\
            \vdots \\
            \tilde{X}_{\\hat{m}} \\mathbf{G}_{\\hat{m}}
            \\end{bmatrix} \\in \\mathbb{R}^{n \times \\hat{m}}
            \\end{equation}
            \\end{align*}

        Parameters
        ----------
        Xs_tilde : BlockMatrix
            The proxy data with shape (c, d).
        Xancs_tilde : BlockMatrix
            The ancillary data with shape (c, d).
        durations : Array1D
            The duration time vector with shape (n,).
        events : Array1D
            The one-hot event vector with shape (n,).
            e.g. 1 for dead, 0 for survived.
        """
        Gs = self._compute_target_matrix(Xancs_tilde)
        self._cox_regression(Xs_tilde, Gs, durations, events)
        return self


@validate_methods
class SurvivalFunction:
    r"""Cox survival function.

    This class provides methods for survival analysis based on the Cox proportional
    hazards model. It supports two centering strategies for baseline hazard computation.

    Centering Strategies
    ---------------------
    The `centering` parameter controls how the baseline hazard and partial hazard
    are computed:

    **centering=None (default, paper convention)**:
        - Baseline hazard is defined at x=0:
          h(t|x) = h_0(t) * exp(x @ beta)
        - The baseline hazard from lifelines (which is at the mean) is transformed:
          h_0_paper(t) = h_0_lifelines(t) * exp(-mean @ beta)
        - Partial hazard: exp(X @ beta)
        - This follows the standard textbook/paper definition where h_0(t) represents
          the hazard when all covariates are zero.

    **centering="mean" (lifelines convention)**:
        - Baseline hazard is defined at x=mean:
          h(t|x) = h_0(t) * exp((x - mean) @ beta)
        - The baseline hazard is used as-is from lifelines.
        - Partial hazard: exp((X - mean) @ beta)
        - This is the convention used by lifelines library, where h_0(t) represents
          the hazard at the mean covariate values.

    Both conventions yield identical hazard predictions h(t|x), but differ in how
    the baseline hazard h_0(t) is defined and reported.
    """

    def __init__(
        self,
        coef: pd.Series,
        coef_var: Array2D,
        baseline_hazard: pd.DataFrame,
        mean: Array1D,
        *,
        alpha: float = 0.05,
        centering: Literal["mean"] | None = None,
    ) -> None:
        self.__coef = coef
        self.__coef_var = coef_var
        self.__mean = np.asarray(mean, dtype=float)
        self.alpha = alpha
        self.centering = centering

        if centering is None:
            # h0_paper(t) = h0_lifelines(t) * exp(-mean^T beta)
            # Transform baseline hazard from "at mean" to "at zero"
            beta_vec = np.asarray(coef.to_numpy(), dtype=float)
            shift = float(self.__mean @ beta_vec)
            scale = float(np.exp(-shift))
            self.__baseline_hazard = baseline_hazard.copy()
            col0 = self.__baseline_hazard.columns[0]
            self.__baseline_hazard[col0] = self.__baseline_hazard[col0] * scale
        else:
            # centering="mean": use baseline hazard as-is (at mean)
            self.__baseline_hazard = baseline_hazard

    @property
    def baseline_cumhazards(self) -> pd.DataFrame:
        r"""
        Baseline cumulative hazards.

        .. math::
            H\\left(t\right)=\\int_0^t h\\left(z\right)dz
        """
        return pd.DataFrame(
            {
                "baseline cumhazards": [
                    self.cumhazard_at(t) for t in self.__baseline_hazard.index
                ]
            },
            index=self.__baseline_hazard.index,
        )

    @property
    def baseline_survival(self) -> pd.DataFrame:
        r"""
        Baseline survivals.

        .. math::
            S\\left(t\right)=\\exp\\left(-H\\left(t\right)\right)
        """
        return pd.DataFrame(
            {
                "baseline survival": [
                    self.survival_at(t) for t in self.__baseline_hazard.index
                ]
            },
            index=self.__baseline_hazard.index,
        )

    @property
    def baseline_hazard(self) -> pd.DataFrame:
        r"""
        Baseline hazards.

        .. math::
            h_0\\left(t\right)
        """
        return self.__baseline_hazard

    @property
    def coef(self) -> pd.DataFrame:
        r"""
        Coefficients.

        .. math::
            \beta
        """
        return self.__coef

    @property
    def coef_var(self) -> Array2D:
        r"""
        Variance-covariance matrix.

        .. math::
            \\mathrm{var}\\left(\beta\right)
        """
        return self.__coef_var

    @property
    def SE(self) -> pd.Series:
        r"""
        Stanadard error.

        .. math::
            \\sqrt{\\mathrm{diag}\\left(\\mathrm{var}\\left(\beta\right)\right)}
        """
        coef_var_ = (
            np.diag(self.__coef_var) if self.__coef_var.ndim == 2 else self.__coef_var
        )
        return pd.Series(np.sqrt(coef_var_), index=self.__coef.index)

    @property
    def stats(self) -> pd.Series:
        r"""
        t-statistics.

        .. math::
            \frac{\beta}{\\mathrm{SE}}
        """
        return self.__coef / self.SE

    @property
    def pvals(self) -> pd.Series:
        r"""
        P-values.

        .. math::
            1-F\\left(|t|^2,\\ 1\right)
        """
        pvals_ = 1 - sp.stats.chi2.cdf(np.square(self.stats), 1)
        return pd.Series(pvals_, index=self.__coef.index)

    @property
    def hazard_ratio(self) -> pd.Series:
        r"""
        Hazard ratios.

        .. math::
            \frac{h\\left(t, \\mathbf{x}+\\mathbf{e}_i\right)}{h\\left(t, \\mathbf{x}\right)} = \\exp(\beta_i)
        """
        return pd.Series(np.exp(self.__coef), index=self.__coef.index)

    @property
    def CI(self) -> pd.DataFrame:
        r"""
        Confidence intervals.

        .. math::
            \begin{align*}
            \beta \\pm Z_{\alpha /2}\\mathrm{SE}
            \\exp\\left(\beta \\pm Z_{\alpha /2}\\mathrm{SE}\right)
            \\end{align*}
        """
        alpha_stats_ = sp.stats.norm.ppf(1 - (self.alpha / 2))
        upper_ = self.__coef + alpha_stats_ * self.SE
        lower_ = self.__coef - alpha_stats_ * self.SE
        ci_name = f"{100 * (1 - self.alpha)}% CI"
        ci_result = {
            f"coef lower {ci_name}": lower_,
            f"coef upper {ci_name}": upper_,
            f"HR lower {ci_name}": np.exp(lower_),
            f"HR upper {ci_name}": np.exp(upper_),
        }
        return pd.DataFrame(ci_result, index=self.__coef.index)

    @property
    def summary(self) -> pd.DataFrame:
        r"""
        Statistical summary.

        Returns
        -------
        pd.DataFrame
            Statistical summary.
        """
        df1 = pd.DataFrame(
            {
                "coef": self.__coef,
                "HR exp(coef)": self.hazard_ratio,
                "SE": self.SE,
            }
        )
        df2 = pd.DataFrame(
            {
                "z": self.stats,
                "p-value": self.pvals,
                "-log2(p)": -np.log2(self.pvals),
                "-log10(p)": -np.log10(self.pvals),
            }
        )
        return pd.concat([df1, self.CI, df2], axis=1)

    def predict_log_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        r"""Log-partial hazard.

        Computes the log-partial hazard based on the centering strategy:

        - centering=None: X @ beta (paper convention)
        - centering="mean": (X - mean) @ beta (lifelines convention)

        The paper uses the standard Cox form h(t|x) = h_0(t) * exp(x^T beta);
        mean-centering is an implementation option to match lifelines' reported
        baseline_hazard_, not a paper requirement.

        Both yield identical final hazard predictions when combined with their
        respective baseline hazards.
        """
        X_ = X.loc[:, self.__coef.index]
        if self.centering == "mean":
            X_ = X_ - self.__mean
        return X_ @ self.__coef

    def predict_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        r"""
        Partial hazard.

        .. math::
            \\exp\\left(\\mathbf{X}\beta\right)
        """
        return np.exp(self.predict_log_partial_hazard(X))

    def hazard_at(self, t: int | float) -> float:
        r"""
        Hazard at time t.

        .. math::
            h_0\\left(t\right)
        """
        # h0(t)
        # TODO: Add Interpolation
        return self.__baseline_hazard.loc[t][0]

    def cumhazard_at(self, t: int | float) -> float:
        r"""
        Cumulative hazard at time t.

        .. math::
            H\\left(t\right)=\\int_0^t h\\left(z\right)dz
        """
        # H(t)
        # TODO: Add Interpolation
        times = [t_ for t_ in self.__baseline_hazard.index if t_ <= t]
        hazards_ = [self.hazard_at(t_) for t_ in times]
        return sum(hazards_, 0.0)

    def survival_at(self, t: int | float) -> float:
        r"""
        Survival at time t.

        .. math::
            S\\left(t\right)=\\exp\\left(-H\\left(t\right)\right)
        """
        # S(t)
        return np.exp(-self.cumhazard_at(t))

    def predict_hazard_at(self, t: int | float, X: pd.DataFrame) -> pd.Series:
        r"""
        Predict hazard at time t.

        .. math::
            h\\left(t,\\ \\mathbf{X}\right) = h_0\\left(t\right)\\exp\\left(\\mathbf{X}\beta\right)
        """
        return self.hazard_at(t) * self.predict_partial_hazard(X)

    def predict_cumhazard_at(self, t: int | float, X: pd.DataFrame) -> pd.Series:
        r"""
        Predict cumulative hazard at time t.

        .. math::
            H\\left(t,\\ \\mathbf{X}\right)=\\int_0^{t}h\\left(z, \\mathbf{X}\right)dz
        """
        return self.cumhazard_at(t) * self.predict_partial_hazard(X)

    def predict_survival_at(self, t: int | float, X: pd.DataFrame) -> pd.Series:
        r"""
        Predict survival probability at time t.

        .. math::
            S\\left(t,\\ \\mathbf{X}\right) = \\exp\\left(-H\\left(t,\\ \\mathbf{X}\right)\right)
        """
        return np.exp(-self.predict_cumhazard_at(t, X))

    def _predict_times(self, func_name: str, X: pd.DataFrame) -> pd.DataFrame:
        pred_ = []
        pred_func = getattr(self, func_name)
        for t in self.__baseline_hazard.index:
            pred_.append(np.array(pred_func(t, X)).reshape(1, -1))
        return pd.DataFrame(np.concatenate(pred_), index=self.__baseline_hazard.index)

    def predict_hazard(self, X: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predict hazard at a series of times.

        .. math::
            h\\left(t,\\ \\mathbf{X}\right) = h_0\\left(t\right)\\exp\\left(\\mathbf{X}\beta\right)
        """
        return self._predict_times("predict_hazard_at", X)

    def predict_cumhazard(self, X: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predict cumulative hazard at a series of times.

        .. math::
            H\\left(t,\\ \\mathbf{X}\right)=\\int_0^{t}h\\left(z, \\mathbf{X}\right)dz
        """
        return self._predict_times("predict_cumhazard_at", X)

    def predict_survival(self, X: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predict survival probability at a series of times.

        .. math::
            S\\left(t,\\ \\mathbf{X}\right) = \\exp\\left(-H\\left(t,\\ \\mathbf{X}\right)\right)
        """
        return self._predict_times("predict_survival_at", X)

    def predict_expectation(self, X: pd.DataFrame) -> Array1D:
        r"""
        Predict expectation of survival time.

        .. math::
            \\mathbb{E}\\left[T\right] = \\int_0^{\\mathrm{inf}}S\\left(t,\\ \\mathbf{X}\right)dt
        """
        survivals_ = self.predict_survival(X)
        return trapz(survivals_.T, survivals_.index)
