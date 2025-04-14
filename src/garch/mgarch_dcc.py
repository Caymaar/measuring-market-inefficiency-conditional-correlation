from .abstract_mgarch import MultivariateGARCH
import numpy as np


class DCC(MultivariateGARCH):
    """
    Class for the DCC (Dynamic Conditional Correlation) model.
    """

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the DCC model to the data.

        Parameters:
            data (np.ndarray): The input data to fit the model.
        """
        pass

    def get_cc_matrix(self) -> np.ndarray:
        """
        Get the conditional covariance matrix over time (3D array).

        Returns:
            np.ndarray: The conditional covariance matrix over time.
        """
        pass