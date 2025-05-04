import numpy as np
from typing import List
from ..lib_hurst.AddMethods import AddMethods
from .abstract_estimation import AbstractHurstEstimator


class HiguchiHurstEstimator(AbstractHurstEstimator, AddMethods):
    """
    Hurst exponent estimator using the Higuchi method.
    """

    def __init__(self, time_series: np.ndarray, k_max: int = 11,
                 method: str = 'L2'):
        """
        Initialize the Higuchi estimator.

        Parameters
        ----------
        time_series : np.ndarray
            The time series data to analyze (fBm or similar).
        k_max : int, optional
            Maximum k value to compute the curve length (default is 11).
        method : str, optional
            Curve fitting method, either 'L2' or 'L1' (default is 'L2').
        """
        super().__init__(time_series)
        self.k_max = k_max
        self.method = method

    def estimate(self) -> float:
        """
        Estimate the Hurst exponent using the Higuchi method.

        Returns
        -------
        float
            The estimated Hurst exponent.
        """
        N = len(self.ts)
        Lk = []
        k_values = range(1, self.k_max)

        for k in k_values:
            Lmk = []
            for m in range(k):
                Lm = 0
                n_max = int(np.floor((N - m - 1) / k))
                for i in range(1, n_max):
                    Lm += abs(self.ts[m + i * k] - self.ts[m + (i - 1) * k])
                if n_max > 0:
                    norm_factor = (N - 1) / (n_max * k)
                    Lmk.append(Lm * norm_factor)
            if Lmk:
                Lk.append(np.mean(Lmk))

        slope = self._fit_log_log(list(k_values), Lk)
        fractal_dim = -slope
        hurst = 2 - fractal_dim
        return hurst

    def _fit_log_log(self, scales: List[int], values: List[float]) -> float:
        """
        Fit a line to the log-log relationship of scales and values.

        Parameters
        ----------
        scales : List[int]
            List of k values.
        values : List[float]
            Corresponding average curve lengths.

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
