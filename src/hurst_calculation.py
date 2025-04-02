import numpy as np
import pandas as pd

class HurstCalculator:
    def __init__(self, k=10, window=250):
        self.k = k
        self.window = window
    
    def get_hurst_exponent(self, time_series):
        """Returns the Hurst Exponent of the time series"""
        lags = range(2, self.k)
        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    
    def rolling_hurst(self, prices: pd.Series) -> pd.Series:
        """
        Calcule l'exposant de Hurst en appliquant une fenêtre glissante sur les rendements logarithmiques.
        
        Paramètres:
            prices: pd.Series, série de prix indexée par date.
        
        Retourne:
            pd.Series des exposants de Hurst, indexée par la date correspondant à la fin de chaque fenêtre.
        """
        # Calcul des rendements logarithmiques et élimination de la première valeur NaN
        log_prices = np.log(prices)
        hurst_values = []
        index_list = []
        
        # Parcourir les fenêtres glissantes
        for i in range(len(log_prices) - self.window + 1):
            window_data = log_prices.iloc[i:i+self.window].values
            h = self.get_hurst_exponent(window_data)
            hurst_values.append(h)
            # L'index associé correspond à la date du dernier élément de la fenêtre
            index_list.append(log_prices.index[i+self.window - 1])
        
        return pd.Series(hurst_values, index=index_list)
    
    def calculate_inefficiency(self, prices: pd.Series) -> pd.Series:
        """
        Calcule l'inefficience de marché en soustrayant l'exposant de Hurst à 0.5.
        
        Paramètres:
            prices: pd.Series, série de prix indexée par date.
        
        Retourne:
            pd.Series des inefficiences, indexée par la date correspondant à la fin de chaque fenêtre.
        """
        hurst_series = self.rolling_hurst(prices)
        inefficiency_series = 0.5 - hurst_series
        return inefficiency_series