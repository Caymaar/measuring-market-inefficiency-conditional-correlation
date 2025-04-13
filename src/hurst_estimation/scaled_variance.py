from .abstract_estimation import AbstractHurstEstimator
from scipy.stats import linregress
import numpy as np


class ScaledVariance(AbstractHurstEstimator):
    """
    Description de la methode.
    """

    def __init__(self, time_series: np.ndarray, max_scale: int = 100):
        """
        Parameters:
            time_series (np.ndarray): The price time series data to analyze.
            max_scale (int): The maximum number of scale to consider.
        """
        self.ts = time_series
        self.max_scale = max_scale

    def estimate(self):
        """
        Estimate the Hurst exponent from the provided data with the scaled variance method.

        Returns:
            float: The estimated Hurst exponent.
        """
        log_price = np.log(self.ts)

        scales = np.arange(1, self.max_scale)
        sigmas = [np.std(np.subtract(log_price[lag:], log_price[:-lag])) for lag in scales]

        slope, _ = linregress(np.log(scales),  np.log(sigmas))

        return slope
