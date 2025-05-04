
import numpy as np
from typing import List
from ..lib_hurst.AddMethods import AddMethods
from .abstract_estimation import AbstractHurstEstimator


class AbsoluteMomentsHurstEstimator(AbstractHurstEstimator, AddMethods):
    """
    Hurst exponent estimator using the Absolute Moments method.
    """

    def __init__(self, time_series: np.ndarray, minimal: int = 20,
                 method: str = 'L2'):
        """
        Initialize the Absolute Moments estimator.

        Parameters
        ----------
        time_series : np.ndarray
            The time series data to analyze.
        minimal : int, optional
            Minimum aggregation scale (default is 20).
        method : str, optional
            Curve fitting method, either 'L2' or 'L1' (default is 'L2').
        """
        super().__init__(time_series)
        self.minimal = minimal
        self.method = method

    def estimate(self) -> float:
        """
        Estimate the Hurst exponent from the provided data using
        the Absolute Moments method.

        Returns
        -------
        float
            The estimated Hurst exponent.
        """
        N = len(self.ts)
        opt_n = self.findOptN(N, minimal=self.minimal)
        scales = self.Divisors(opt_n, minimal=self.minimal)
        ts_segment = self.ts[N - opt_n:]
        mean_ts = np.mean(ts_segment)
        moments: List[float] = []
        for m in scales:
            k = opt_n // m
            subs = np.reshape(ts_segment, (k, m))
            block_means = np.mean(subs, axis=1)
            moments.append(np.linalg.norm(block_means - mean_ts, 1) / len(block_means))
        slope = self._fit_log_log(scales, moments)
        return slope + 1

    def _fit_log_log(self, scales: List[int],
                     values: List[float]) -> float:
        """
        Fit a line to the log-log relationship of scales and values.

        Parameters
        ----------
        scales : List[int]
            Aggregation scales.
        values : List[float]
            Corresponding absolute moment statistics.

        Returns
        -------
        float
            Slope of the fitted line in log-log space.
        """
        X = np.vstack([np.log10(scales), np.ones(len(values))]).T
        y = np.log10(values)
        if self.method == 'L2':
            slope, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            slope = self.OLE_linprog(X, y.reshape(-1, 1))[0]
        return slope
