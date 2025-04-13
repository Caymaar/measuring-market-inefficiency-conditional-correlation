import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enums import HurstMethodType

class InefficiencyCalculator:
    """
    Class to calculate the inefficiency of a time series using the Hurst exponent.
    """
    def __init__(self, hurst_method_type: HurstMethodType, params: dict):
        """
        Initialize the InefficiencyCalculator with the Hurst method type and parameters.
        Parameters:
            hurst_method_type (HurstMethodType): Hurst estimation method enum
            params (dict): Parameters for the Hurst calculation
        """

        self.params = params
        self.window = params.get('window', 100)
        self.hurst_method_type = hurst_method_type.value

    def _rolling_hurst(self, time_series: pd.Series) -> pd.Series:
        """
        Calculate rolling Hurst exponent using specified method
        Parameters:
            time_series: Time series indexed by date
        Returns:
            Series of Hurst exponents
        """

        hurst_values = []
        index_list = []
        
        for i in range(len(time_series) - self.window + 1):
            window = time_series.iloc[i:i+self.window].values

            hurst_method = self.hurst_method_type(window, *self.params['method_params'])
            h = hurst_method.estimate()

            hurst_values.append(h)
            index_list.append(time_series.index[i+self.window - 1])    

        return pd.Series(hurst_values, index=index_list)      

    def calculate_inefficiency(self, time_series: pd.Series) -> pd.Series:
        """
        Calculate market inefficiency based on the Hurst exponent.
        Parameters:
            time_series: Time series indexed by date
        Returns:
            Series of inefficiency values
        """

        hurst_series = self._rolling_hurst(time_series)
        return 0.5 - hurst_series
