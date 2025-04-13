from typing import Dict, List
import pandas as pd
from enums import HurstMethodType
import utils


class Launcher:
    """
    Run the Framework for each set of dates.
    """

    def __init__(self, data: List[str], dates: Dict[str, List[str]], hurst_method: HurstMethodType, params=Dict[str: any]):
        """
        Parameters:
            data (List[str]): Input data names
            dates (Dict[str, list[str]]): Set of dates for a given period format : Period_Name = Period_Dates
            hurst_method (HurstMethodType): Hurst estimation method enum
            params (Dict[str: any]): global parameters
        """
        start_date = min([pd.to_datetime(period['start_date']) for period in dates.values()])
        end_date = max([pd.to_datetime(period['end_date']) for period in dates.values()])

        self.global_data = {undl: utils.get_data(undl, start_date, end_date) for undl in data}
    
    def run_process(self):
        """
        Run the full process for each set of date.
        """

        for period_name, set_dates in self.dates.items():
            print(f"Processing period: {period_name} with dates: {set_dates}")
            period_data = {
                undl: data[(data.index >= start_date) & (data.index <= end_date)]
                for undl, data in self.global_data.items()
            }
            framework = Framework(data, self.hurst_method, self.params)
            framework.run()



if __name__ == "__main__":

    # Example usage
    data = ["AAPL", "GOOGL", "MSFT"]
    dates = {
        "2023-01": ["2023-01-01", "2023-01-31"],
        "2023-02": ["2023-02-01", "2023-02-28"]
    }
    hurst_method = HurstMethodType.CLASSIC
    params = {"window": 30}

    launcher = Launcher(data, dates, hurst_method, params)
    launcher.run_process()