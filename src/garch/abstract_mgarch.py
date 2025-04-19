from abc import ABC, abstractmethod
import numpy as np


class MultivariateGARCH(ABC):
    """
    Abstract class for multivariate GARCH models.
    """

    def __init__(self, p: int, q: int) -> None:
        """
        Initialize the Multivariate GARCH model.

        Parameters:
            p (int): The order of the GARCH model.
            q (int): The order of the ARCH model.
        """
        self.p = p
        self.q = q

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Fit the multivariate GARCH model to the data.

        Parameters:
            data (np.ndarray): The input data to fit the model.
        """
        pass

    @abstractmethod
    def get_cc_matrix(self) -> np.ndarray:
        """
        Get the conditional correlation matrix over time (3D array).

        Returns:
            np.ndarray: The conditional correlation matrix over time.
        """
        pass