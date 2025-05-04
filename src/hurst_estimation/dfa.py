import numpy as np
from typing import List
from ..lib_hurst.AddMethods import AddMethods
from .abstract_estimation import AbstractHurstEstimator


class DfaHurstEstimator(AbstractHurstEstimator, AddMethods):
    """
    Hurst exponent estimator using Detrended Fluctuation Analysis (DFA).
    """

    def __init__(self, time_series: np.ndarray, minimal: int = 20,
                 method: str = 'L2'):
        """
        Initialize the DFA estimator.

        Parameters
        ----------
        time_series : np.ndarray
            The time series data to analyze (should be a fBm, not fGn).
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
        the DFA method.

        Returns
        -------
        float
            The estimated Hurst exponent.
        """
        N = len(self.ts)
        opt_n = self.findOptN(N, minimal=self.minimal)
        scales = self.Divisors(opt_n, minimal=self.minimal)

        # Do NOT apply cumulative sum again: fBm is already integrated
        X = self.ts[N - opt_n:] - np.mean(self.ts[N - opt_n:])

        fluctuations: List[float] = []
        for m in scales:
            k = opt_n // m
            subs = np.reshape(X[:k * m], (k, m))
            flucts = []
            for segment in subs:
                t = np.arange(m)
                coeffs = np.polyfit(t, segment, deg=1)
                trend = np.polyval(coeffs, t)
                flucts.append(np.mean((segment - trend) ** 2))
            fluctuations.append(np.sqrt(np.mean(flucts)))

        slope = self._fit_log_log(scales, fluctuations)
        return slope

    def _fit_log_log(self, scales: List[int], values: List[float]) -> float:
        """
        Fit a line to the log-log relationship of scales and fluctuation values.

        Parameters
        ----------
        scales : List[int]
            Aggregation scales.
        values : List[float]
            Fluctuation values at each scale.

        Returns
        -------
        float
            Slope of the fitted line in log-log space, representing the Hurst exponent.
        """
        X = np.vstack([np.log10(scales), np.ones(len(values))]).T
        y = np.log10(values)
        if self.method == 'L2':
            slope, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            slope = self.OLE_linprog(X, y.reshape(-1, 1))[0]
        return slope
