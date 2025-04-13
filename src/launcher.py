from typing import Dict, List, Tuple
import pandas as pd
from .framework import Framework
from .enums import HurstMethodType
from src import utils


class Launcher:
    """
    Run the Framework for each set of dates.
    """

    def __init__(self, data: List[str], dates: Dict[str, List[str]], hurst_method: HurstMethodType, params=Dict[str, any]):
        """
        Parameters:
            data (List[str]): Input data names
            dates (Dict[str, list[str]]): Set of dates for a given period format : Period_Name = Period_Dates
            hurst_method (HurstMethodType): Hurst estimation method enum
            params (Dict[str: any]): global parameters
        """
        self.hurst_method = hurst_method
        self.dates = dates
        self.params = params

        start_date = min([pd.to_datetime(period[0], dayfirst=True) for period in dates.values()]) - pd.DateOffset(days=self.params.get('window', 0))
        end_date = max([pd.to_datetime(period[1], dayfirst=True) for period in dates.values()])

        global_data = {undl: utils.get_data(undl, start_date, end_date) for undl in data}

        self.df = self._to_dataframe(global_data)
    
    def _get_sub_dataframe(self, dates: Tuple[str, str]) -> Dict[str, pd.Series]:
        """
        Create the sub series for a given period.

        Parameters:
            dates (Tuple[str, str]): Start and end date of the period

        Returns:
            Dict[str, pd.Series]: Sub series for the period
        """

        start_date, end_date = pd.to_datetime(dates, dayfirst=True)

        # Trouver l'index de la date la plus proche (remplissage vers l'avant)
        start_date_index = self.df.index.get_indexer([start_date], method='ffill')[0]

        # Calculer l'index de la date qui est self.params['window'] jours avant
        start_date_index = max(0, start_date_index - self.params.get('window', 0))
        start_date = self.df.index[start_date_index]

        # Filter the DataFrame to get the sub-series
        return self.df.loc[start_date:end_date]
    
    def _to_dataframe(self, data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Transform the data into a DataFrame.

        Parameters:
            data (Dict[str, pd.Series]): Input data

        Returns:
            pd.DataFrame: DataFrame with the data
        """

        common_index = pd.date_range(
            start=max(series.index.min() for series in data.values()),
            end=min(series.index.max() for series in data.values()),
            freq='B',
        )

        full_sample = pd.DataFrame({
            undl: series.reindex(common_index)
            for undl, series in data.items()
        })

        return full_sample.ffill()
    
    def run_process(self):
        """
        Run the full process for each set of date.
        """
        for period_name, set_dates in self.dates.items():
            print(f"Processing period: {period_name} with dates: {set_dates}")
            period_data = self._get_sub_dataframe(set_dates)
            framework = Framework(period_data, self.hurst_method, self.params)
            framework.run()
