import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import solve
from typing import Tuple
from scipy.stats import linregress
from src.ext.HurstIndexSolver import HurstIndexSolver
from src.hurst_methods import ScaledWindowedVariance

class HurstCalculator:
    def __init__(self, k: int = 10, window: int = 250, method: str = 'swv'):
        """
        Initialize the Hurst Calculator
        
        Parameters:
            k: Number of lags to use
            window: Size of rolling window
            method: 'classic' or 'penalized' regression approach
        """
        self.k = k
        self.window = window
        self.method = method
        self.HIS = HurstIndexSolver()

    def get_rs_hurst(self, log_returns):
        """
        Calculates the Hurst Exponent using the Rescaled Range (R/S) analysis method.
        """
        # Compute log returns
        #log_returns = np.diff(np.log(price_series))
        
        # Create an array of lag values
        lags = range(2, self.k)
        
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(log_returns[lag:], log_returns[:-lag]))) for lag in lags]
        
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # The Hurst exponent is the slope of the linear fit
        hurst_exponent = poly[0]*2.0
        
        # The fractal dimension is related to the Hurst exponent
        fractal_dimension = 2 - hurst_exponent
        
        return hurst_exponent, fractal_dimension 
    
    def estimate_hurst(self, prix: np.ndarray, max_tau: int = 100, plot: bool = False) -> float:
        """
        Estime l'exposant de Hurst à partir d'une série de prix.

        Parameters:
            log_prix (np.ndarray): Série de prix.
            max_tau (int): Nombre maximum de décalages (tau) à considérer.
            plot (bool): Si True, affiche la régression log-log.

        Returns:
            float: L'exposant de Hurst estimé.
        """
        log_prix = np.log(prix)
        
        # Calcul des sigmas pour chaque tau
        taus = np.arange(1, max_tau)
        sigmas = [np.std(np.subtract(log_prix[lag:], log_prix[:-lag])) for lag in taus]

        # Régression linéaire sur les valeurs log-log
        log_echelles = np.log(taus)
        log_sigma_t = np.log(sigmas)
        slope, intercept, _, _, _ = linregress(log_echelles, log_sigma_t)
        return slope

    def get_matlab_hurst(self, Signal):
        """
        Calculate the Hurst exponent using the Aggregated Variance Method.
        
        Parameters
        ----------
        Signal : array-like
            1D time series.
            
        Returns
        -------
        hur : float
            Estimated Hurst exponent.
            - hur <= 0.5: the time series is noise.
            - hur > 0.5: the time series exhibits long-term memory (a trend is present).
        """
        # Ensure the signal is a 1D numpy array
        Signal = np.asarray(Signal).flatten()
        N = len(Signal)
        
        var_list = []
        m_list = []
        
        m = 2
        while m <= N/2:
            # Only use m that divides N exactly
            if N % m != 0:
                m += 1
                continue
            k = N // m  # number of segments
            
            means = np.zeros(k)
            for i in range(k):
                segment = Signal[i*m:(i+1)*m]
                means[i] = np.mean(segment)
            
            # Using sample variance (ddof=1) to mimic MATLAB's var behavior
            var_list.append(np.var(means, ddof=1))
            m_list.append(m)
            m += 1

        M = np.array(m_list)
        Var = np.array(var_list)
        
        # Construct the design matrix for linear regression:
        # log(Var) = hX[0] * log(M) + hX[1]
        A = np.column_stack((np.log(M), np.ones(len(M))))
        logVar = np.log(Var)
        
        # Solve the least-squares problem
        hX, _, _, _ = np.linalg.lstsq(A, logVar, rcond=None)
        
        # Compute the Hurst exponent:
        # hur = hX[0]/2 + 1
        hur = hX[0] / 2 + 1
        return hur
        
    def get_penalized_hurst(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using penalized regression approach
        
        Parameters:
            time_series: Input time series
            
        Returns:
            Estimated Hurst exponent
        """
        lags = range(2, self.k)
        # Calculate variances of lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        # Prepare design matrix X
        log_lags = np.log(lags)
        X = sm.add_constant(log_lags)
        y = np.log(tau)
        
        # Define penalty matrix S (second-order difference penalty) @13
        n = X.shape[1]
        S = np.zeros((n, n))
        S[1,1] = 1  # Penalty only on slope parameter
        
        # Estimate optimal smoothing parameter using GCV @14
        def gcv(lambda_: float) -> float:
            # Penalized least squares estimation
            beta_hat = solve(X.T @ X + lambda_ * S, X.T @ y)
            fitted = X @ beta_hat
            n = len(y)
            A = X @ solve(X.T @ X + lambda_ * S, X.T)
            df = np.trace(A)
            residuals = y - fitted
            gcv = n * np.sum(residuals**2) / (n - df)**2
            return gcv
        
        # Find optimal lambda using grid search
        lambdas = np.logspace(-3, 3, 100)
        gcv_scores = [gcv(lambda_) for lambda_ in lambdas]
        optimal_lambda = lambdas[np.argmin(gcv_scores)]
        
        # Final estimation with optimal lambda @13
        beta_hat = solve(X.T @ X + optimal_lambda * S, X.T @ y)
        return beta_hat[1]  # Return slope coefficient
    
    def get_classic_hurst(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using classic regression approach
        
        Parameters:
            time_series: Input time series
            
        Returns:
            Estimated Hurst exponent
        """
        lags = range(2, self.k)
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    def get_swv_hurst(self, time_series: np.ndarray) -> float:

        SWV = ScaledWindowedVariance(time_series)
        slope, _ = SWV.estimate_hurst(method='SD', exclusions=True)
        

    def rolling_hurst(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Hurst exponent using specified method
        
        Parameters:
            prices: Price series indexed by date
            
        Returns:
            Series of Hurst exponents
        """
        log_prices = np.log(prices)
        diff_prices = np.diff(log_prices)
        hurst_values = []
        index_list = []
        
        for i in range(len(log_prices) - self.window + 1):
            window_data = log_prices.iloc[i:i+self.window].values
            window_returns = diff_prices[i:i+self.window]

            
            if self.method == 'penalized':
                h = self.get_penalized_hurst(window_data)
            elif self.method == 'matlab':
                h = self.get_matlab_hurst(window_data)
            elif self.method == 'rs':
                h = self.get_rs_hurst(window_returns)[0]
            elif self.method == 'HIS':
                h = self.HIS.EstHurstAggregateVariance(window_data, minimal=self.k, method="L2")
            elif self.method == 'thibz':
                h = self.estimate_hurst(window_data, max_tau=self.k)
            elif self.method == 'swv':
                h = self.get_swv_hurst(window_data)
            else:
                h = self.get_classic_hurst(window_data)

                
            hurst_values.append(h)
            index_list.append(log_prices.index[i+self.window - 1])
            
        return pd.Series(hurst_values, index=index_list)

    def calculate_inefficiency(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate market inefficiency and volatility measures
        
        Parameters:
            prices: Price series indexed by date
            
        Returns:
            Tuple of (inefficiency series, conditional volatility series)
        """
        # Calculate Hurst-based inefficiency @13, @14
        hurst_series = self.rolling_hurst(prices)
        inefficiency_series = 0.5 - hurst_series
        
        # Calculate conditional volatility using H-GARCH approach @7
        log_returns = np.log(prices/prices.shift(1)).dropna()
        garch_model = sm.regression.linear_model.OLS(
            inefficiency_series,
            sm.add_constant(log_returns[inefficiency_series.index])
        ).fit()
        
        return inefficiency_series, pd.Series(garch_model.resid**2, index=inefficiency_series.index)