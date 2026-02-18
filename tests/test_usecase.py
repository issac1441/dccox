"""Test cases for DC-Cox."""

from __future__ import annotations

import os
import shutil
import unittest

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import numpy as np
import pandas as pd
import pytest

from dccox.usecase import Horizontal


class CoxPHRegressionTestCase(unittest.TestCase):
    """Test case for CoxPHRegression."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures before running tests in the class."""
        cls.data = load_rossi()
        cls.data.columns = ["time", "event", *cls.data.columns[2:]]
        training_size = 400
        cls.ans = CoxPHFitter()
        cls.ans.fit(
            cls.data.iloc[0:training_size, :], event_col="event", duration_col="time"
        )
        cls.keep_feature_cols = list(cls.data.columns[2:])

        # Generate testing data
        cls.test_dir = "/tmp/_test_cox"
        cls.file_paths = []
        if not os.path.exists(cls.test_dir):
            os.mkdir(cls.test_dir)
        idx = np.random.choice(training_size, training_size, replace=False)
        cls.n_chunk = 4
        chunk_size = len(idx) // cls.n_chunk
        for i in range(cls.n_chunk):
            X_ = cls.data.iloc[i * chunk_size : (i + 1) * chunk_size, 2:]
            y_ = cls.data.iloc[i * chunk_size : (i + 1) * chunk_size, 0:2]
            data_ = pd.concat([X_, y_], axis=1)
            filepath = f"{cls.test_dir}/client{i + 1}.clinical"
            data_.to_csv(filepath, index=None)
            cls.file_paths.append(filepath)

        cls.dccox = Horizontal()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test fixtures after running tests in the class."""
        shutil.rmtree(cls.test_dir)

    def test_missing_time_and_event(self) -> None:
        """Test that missing time and event columns raise an assertion error."""
        fake = pd.DataFrame(
            np.random.randn(4, 5), columns=["time"] + [f"f{i}" for i in range(4)]
        )
        filepath = f"{self.test_dir}/fake.clinical"
        fake.to_csv(filepath, index=None)

        with pytest.raises(AssertionError, match="Missing columns \\['event'\\]"):
            self.dccox.local_load_metadata(filepath, keep_feature_cols=["f1", "f2"])

    def test_drop_missings(self) -> None:
        """Test that missing values are dropped correctly."""
        fake = np.array(
            [
                [12, 0, np.nan, 1, 2, 3],
                [13, 1, 4, 5, 6, np.nan],
                [11, 1, 7, np.nan, 8, 9],
                [9, 0, 10, 11, np.nan, 12],
            ]
        )
        fake = pd.DataFrame(
            fake, columns=["time", "event"] + [f"f{i}" for i in range(4)]
        )
        filepath = f"{self.test_dir}/fake2.clinical"
        fake.to_csv(filepath, index=None)

        X, y, _, _ = self.dccox.local_load_metadata(
            filepath, keep_feature_cols=["f0", "f1", "f2", "f3"]
        )
        np.testing.assert_array_equal(X, np.array([]).reshape(-1, 4))
        np.testing.assert_array_equal(y, np.array([]).reshape(-1, 2))

        Xanc = self.dccox.global_create_Xanc(len(["f0", "f1", "f2", "f3"]))
        F, X_tilde, Xanc_tilde, _ = self.dccox.local_create_proxy_data(X, Xanc, y)

        assert F is None
        assert X_tilde == [None]
        assert Xanc_tilde == [None]

    def test_dccox(self) -> None:
        """Test the complete DCCOX pipeline."""
        # Generate the global anchor matrix
        Xanc = self.dccox.global_create_Xanc(len(self.keep_feature_cols))

        # Generate projected proxy matrix
        Xs_tilde, Xancs_tilde, Fs, ys, sums = [], [], [], [], []
        for i in range(self.n_chunk):
            X, y, keep_feature_cols, _ = self.dccox.local_load_metadata(
                self.file_paths[i], keep_feature_cols=self.keep_feature_cols
            )
            F, X_tilde, Xanc_tilde, sum_ = self.dccox.local_create_proxy_data(
                X, Xanc, y
            )
            Xs_tilde.append(X_tilde)
            Xancs_tilde.append(Xanc_tilde)
            Fs.append(F)
            ys.append(y)
            sums.append(sum_)
        # Perform cox ph regression
        coef, coef_var, baseline_hazard, mean = self.dccox.global_fit_model(
            Xs_tilde, Xancs_tilde, ys, sums
        )

        # Recover survival function
        surv_func = self.dccox.local_recover_survival(
            keep_feature_cols, coef[0][0], coef_var[0][0], baseline_hazard, mean, Fs[0]
        )

        # baseline hazard
        pd.testing.assert_frame_equal(
            surv_func.baseline_hazard, self.ans.baseline_hazard_
        )

        # coef
        np.testing.assert_array_almost_equal(
            surv_func.coef.to_numpy(), self.ans.summary["coef"].to_numpy()
        )

        # var-cov matrix
        np.testing.assert_array_almost_equal(
            np.diag(surv_func.coef_var), np.diag(self.ans.variance_matrix_)
        )

        # all statistical variables
        np.testing.assert_array_almost_equal(
            surv_func.summary.iloc[:, 0:10],
            self.ans.summary.iloc[:, [i for i in range(11) if i != 7]],
        )

        # cumulative hazard
        pd.testing.assert_frame_equal(
            surv_func.predict_cumhazard(self.data),
            self.ans.predict_cumulative_hazard(self.data),
        )

        # survival probability
        pd.testing.assert_frame_equal(
            surv_func.predict_survival(self.data),
            self.ans.predict_survival_function(self.data),
        )

        # expected survival days
        np.testing.assert_array_almost_equal(
            surv_func.predict_expectation(self.data),
            self.ans.predict_expectation(self.data).values,
        )

    def test_dccox_with_missing(self) -> None:
        """Test DCCOX with missing data."""
        # Generate missing data
        sim_missing = self.data.copy()
        sim_missing["mar"] = np.nan
        filepath = f"{self.test_dir}/missing.clinical"
        sim_missing.to_csv(filepath, index=None)

        # Generate the global anchor matrix
        Xanc = self.dccox.global_create_Xanc(len(self.keep_feature_cols))

        # Generate projected proxy matrix
        Xs_tilde, Xancs_tilde, Fs, ys, sums = [], [], [], [], []
        file_paths = [*self.file_paths, filepath]
        for i in range(self.n_chunk + 1):
            X, y, keep_feature_cols, _ = self.dccox.local_load_metadata(
                file_paths[i], keep_feature_cols=self.keep_feature_cols
            )
            F, X_tilde, Xanc_tilde, sum_ = self.dccox.local_create_proxy_data(
                X, Xanc, y
            )
            Xs_tilde.append(X_tilde)
            Xancs_tilde.append(Xanc_tilde)
            Fs.append(F)
            ys.append(y)
            sums.append(sum_)
        # Perform cox ph regression
        coef, coef_var, baseline_hazard, mean = self.dccox.global_fit_model(
            Xs_tilde, Xancs_tilde, ys, sums
        )

        # Recover survival function
        surv_func = self.dccox.local_recover_survival(
            keep_feature_cols, coef[0][0], coef_var[0][0], baseline_hazard, mean, Fs[0]
        )

        # baseline hazard
        pd.testing.assert_frame_equal(
            surv_func.baseline_hazard, self.ans.baseline_hazard_
        )

        # coef
        np.testing.assert_array_almost_equal(
            surv_func.coef.to_numpy(), self.ans.summary["coef"].to_numpy()
        )

        # var-cov matrix
        np.testing.assert_array_almost_equal(
            np.diag(surv_func.coef_var), np.diag(self.ans.variance_matrix_)
        )

        # all statistical variables
        np.testing.assert_array_almost_equal(
            surv_func.summary.iloc[:, 0:10],
            self.ans.summary.iloc[:, [i for i in range(11) if i != 7]],
        )

        # cumulative hazard
        pd.testing.assert_frame_equal(
            surv_func.predict_cumhazard(self.data),
            self.ans.predict_cumulative_hazard(self.data),
        )

        # survival probability
        pd.testing.assert_frame_equal(
            surv_func.predict_survival(self.data),
            self.ans.predict_survival_function(self.data),
        )

        # expected survival days
        np.testing.assert_array_almost_equal(
            surv_func.predict_expectation(self.data),
            self.ans.predict_expectation(self.data).values,
        )
