from typing import Dict
import pandas as pd
from enums import HurstMethodType


class Framework:
    """
    Framework buit to analyse the conditional correlations & the causality between markets' inefficiencies.
    """

    def __init__(self, data: Dict[str: pd.Series], hurst_method: HurstMethodType, params=Dict[str: any]):
        """
        Instanciate the Framework by setting the input data and the parameters.

        Parameters:
            data (Dict[str: pd.Series]): Input data with the following format : Serie_Name = Serie_Data
            hurst_method (HurstMethodType): Hurst estimation method enum
            params (Dict[str: any]): global parameters
        """
        ...

    def _compute_inefficiency(self):
        """
        Call the InefficiencyCalculator module and launch the rolling estimation of inefficiency.
        """
        self.inefficiency_series = ...

    def _compute_conditional_correlations(self):
        """
        Call the DCC module and launch the dynamic conditional correlation.
        """
        self.dcc_series = ...

    def _compute_granger_causality(self):
        """
        Call the GrangerCausality module and launch the causality estimation throught VAR modeling and Granger causality test.
        """
        self.granger_tests = ...

    def run(self):
        """
        Run the full process for each start_date, end_date.
        """
