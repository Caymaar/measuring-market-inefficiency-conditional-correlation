import numpy as np
from typing import List
from ..lib_hurst.AddMethods import AddMethods
from .abstract_estimation import AbstractHurstEstimator


class TtaHurstEstimator(AbstractHurstEstimator, AddMethods):
    """
    Hurst exponent estimator using time-lag autocorrelation decay.
    """

    def __init__(self, time_series: np.ndarray, max_lag: int = 100, method: str = 'L2'):
        """
        Initialize the TTA estimator.

        Parameters
        ----------
        time_series : np.ndarray
            Time series (assumed to be fBm).
        max_lag : int
            Maximum time lag to use for autocorrelation estimation.
        method : str
            Regression method ('L2' or 'L1').
        """
        super().__init__(time_series)
        self.max_lag = max_lag
        self.method = method

    def estimate(self) -> float:
        """
        Estimate Hurst exponent via autocorrelation decay.

        Returns
        -------
        float
            Estimated Hurst exponent.
        """
        ts = self.ts - np.mean(self.ts)
        N = len(ts)

        autocorrs = []
        lags = range(2, self.max_lag)
        for tau in lags:
            if tau >= N:
                continue
            ac = np.corrcoef(ts[:-tau], ts[tau:])[0, 1]
            if np.isnan(ac) or ac <= 0:
                continue
            autocorrs.append(ac)

        lags = [l for l, ac in zip(lags, autocorrs)]
        autocorrs = [ac for ac in autocorrs]

        if len(lags) < 2:
            raise ValueError("Not enough valid autocorrelation values to estimate H.")

        slope = self._fit_log_log(lags, autocorrs)
        hurst = (slope + 2) / 2
        return hurst

    def _fit_log_log(self, scales: List[int], values: List[float]) -> float:
        X = np.vstack([np.log10(scales), np.ones(len(values))]).T
        y = np.log10(values)
        if self.method == 'L2':
            slope, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            slope = self.OLE_linprog(X, y.reshape(-1, 1))[0]
        return slope
