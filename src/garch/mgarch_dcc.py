from .abstract_mgarch import MultivariateGARCH
import numpy as np
from typing import Tuple, List
from arch import arch_model
from scipy.optimize import minimize
import pandas as pd

class DCC(MultivariateGARCH):
    """
    This module implements the DCC (Dynamic Conditional Correlation) model.

    It uses the following procedure:
        1. Fit univariate GARCH models to each time series in the dataset.
        2. Compute the standardized residuals of the GARCH models.
        3. Estimate the DCC parameters (a, b) using maximum likelihood estimation.
        4. Compute the conditional covariance matrices over time.

    We use the method described in "Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models" by Engle (2002).
    """

    def __init__(self, initial_params: List[float] = [0.05, 0.85], params_bounds: List[Tuple[float]] = [(1e-6, 0.97), (1e-6, 0.97)]) -> None:
        """
        Initialize the DCC model with initial parameters and bounds.

        Parameters:
            initial_params (List[float]): Initial parameters for the DCC model.
            params_bounds (List[Tuple[float]]): Bounds for the DCC parameters.
        """
        self.initial_params = initial_params
        self.params_bounds = params_bounds

        self.T, self.N = None, None
        self.models = {}
        self.a = None
        self.b = None
        self.Q_t = None
        self.R_t = None

    def _fit_univariate_garch(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit univariate GARCH models to each column of the data.
        Compute the standardized residuals of the GARCH models and the unconditional covariance matrix.

        Parameters:
            data (pd.DataFrame): The input data to fit the univariate GARCH models.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The standardized residuals and the unconditional covariance matrix.
        """
        for col in data.columns:
            self.models[col] = arch_model(data[col], p=1, q=1, mean='Zero').fit(disp='off')

        Z = np.vstack([self.models[col].std_resid for col in data.columns]).T
        Q_bar = np.cov(Z, rowvar=False)

        return Z, Q_bar 
    
    @staticmethod
    def _dcc_likelihood(dcc_params: List[float], Z: np.ndarray, Q_bar: np.ndarray) -> float:
        """
        Calculate the likelihood of the DCC model.

        Parameters:
            dcc_params (List[float]): The DCC parameters (a, b).
            Z (np.ndarray): The standardized residuals.
            Q_bar (np.ndarray): The unconditional covariance matrix.

        Returns:
            float: The log-likelihood of the DCC model with the given parameters.
        """
        a, b = dcc_params
        T, N = Z.shape
        Qt = Q_bar.copy()
        log_likelihood = 0.0
        epsilon = 1e-8  # Régularisation

        for t in range(1, T):
            z = Z[t-1].reshape(-1, 1)
            Qt = (1 - a - b) * Q_bar + a * (z @ z.T) + b * Qt
            Qt = 0.5 * (Qt + Qt.T) + epsilon * np.eye(N)

            diag_sqrt = np.sqrt(np.diag(Qt))
            if np.any(diag_sqrt < 1e-10):
                return np.inf

            inv_sqrt = np.diag(1.0 / diag_sqrt)
            Rt = inv_sqrt @ Qt @ inv_sqrt

            try:
                det = np.linalg.det(Rt)
                if det <= 0:
                    return np.inf
                log_likelihood -= 0.5 * (np.log(det) + Z[t] @ np.linalg.solve(Rt, Z[t]))
            except np.linalg.LinAlgError:
                return np.inf

        return log_likelihood

    def _estimate_dcc_parameters(self) -> Tuple[float, float]:
        """
        Estimate the DCC parameters (a, b) using the maximum likelihood estimation.

        Parameters:
            Z (np.ndarray): The standardized residuals.
            Q_bar (np.ndarray): The unconditional covariance matrix.

        Returns:
            Tuple[float, float]: The estimated DCC parameters (a, b).
        """
        constraints = {'type': 'ineq', 'fun': lambda x: 0.99 - 1e-4 - x[0] - x[1]}
        
        result = minimize(
            fun=self._dcc_likelihood,
            x0=self.initial_params,
            args=(self.Z, self.Q_bar),
            method='SLSQP',
            bounds=self.params_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            raise RuntimeError(f"Échec de l'optimisation DCC : {result.message}\nParamètres essayés : {result.x}")
        return result.x
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the DCC model to the data.

        Parameters:
            data (pd.DataFrame): The input data to fit the model.
        """
        if not isinstance(data, pd.DataFrame):
           raise TypeError("Please provide a pandas DataFrame as input data for the DCC model.")
        if data.shape[1] < 2:
            raise ValueError("The DCC model requires at least two time series.")

        self.Z, self.Q_bar = self._fit_univariate_garch(data)

        self.a, self.b = self._estimate_dcc_parameters()

    def get_cc_matrix(self) -> np.ndarray:
        """
        Get the conditional correlation matrix over time (3D array).

        Returns:
            np.ndarray: The conditional correlation matrix over time.
        """
        Qt = self.Q_bar.copy()
        T, N = self.Z.shape
        R_t = np.zeros((T, N, N))
        for t in range(T):
            if t > 0:
                z = self.Z[t-1][:, None]
                Qt = (1 - self.a - self.b) * self.Q_bar + self.a * (z @ z.T) + self.b * Qt
            inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Qt)))
            R_t[t] = inv_sqrt @ Qt @ inv_sqrt

        return R_t