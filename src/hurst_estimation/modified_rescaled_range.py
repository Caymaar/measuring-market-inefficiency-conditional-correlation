from .abstract_estimation import AbstractHurstEstimator
from scipy.stats import linregress
import numpy as np
from .utils_hurst_estimation import ComputeRS


class ModifiedRSMethod(AbstractHurstEstimator):
    """
    Estimation de l’exposant de Hurst par la méthode R/S classique.
    """

    def __init__(self, time_series: np.ndarray, window_size):
        """
        Parameters:
            time_series (np.ndarray or pandas.Series): la série de rendements

        """
        self.time_series = time_series
        self.window_size = window_size


    def estimate(self) -> float:
        """
        Calcule pour chaque échelle s le ratio R/S, puis ajuste une droite
        sur log(R/S) = H * log(t) + b pour en extraire H.

        Returns:
            float: exposant de Hurst estimé.
        """
        rs = ComputeRS.rs_statistic(
            series=self.time_series,
            window_size=self.window_size,
        )
        hurst = np.log(rs) / np.log(self.window_size)

        return hurst

    def rolling_rs(self):
        """
        Calcule le Hurst exponent en utilisant la méthode R/S sur une fenêtre glissante.

        Returns:
            np.ndarray: tableau contenant les valeurs du Hurst exponent.
        """
        hurst_exponents = []
        for i in range(len(self.time_series) - self.window_size + 1):
            window = self.time_series[i:i + self.window_size]
            rs = ComputeRS.rs_statistic(
                series=window,
                window_size=self.window_size,
            )
            hurst = np.log(rs) / np.log(self.window_size)
            hurst_exponents.append(hurst)

        return np.array(hurst_exponents)
