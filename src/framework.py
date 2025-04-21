from typing import Dict
import pandas as pd
from .enums import HurstMethodType, GarchMethodType
from .inefficiency_calculator import InefficiencyCalculator
from .results import Results
from .garch.dcc.dcc_implementation_package_cleaned import mgarch
import matplotlib.pyplot as plt


class Framework:
    """
    Framework buit to analyse the conditional correlations & the causality between markets' inefficiencies.
    """

    def __init__(self, data: pd.DataFrame, hurst_method: HurstMethodType, garch_type: GarchMethodType, params=Dict[str, None]):
        """
        Instanciate the Framework by setting the input data and the parameters.

        Parameters:
            data (Dict[str: pd.Series]): Input data with the following format : Serie_Name = Serie_Data
            hurst_method (HurstMethodType): Hurst estimation method enum
            garch_type (GarchMethodType): GARCH estimation method enum
            params (Dict[str: any]): global parameters
        """
        self.data = data
        self.hurst_method = hurst_method
        self.garch_type = garch_type
        self.params = params

        # Initialize the output series
        self.inefficiency_df = pd.DataFrame()
        self.dcc_df = pd.DataFrame()
        self.granger_tests = {}

    def _compute_inefficiency(self):
        """
        Call the InefficiencyCalculator module and launch the rolling estimation of inefficiency.
        """

        self.inefficiency_calculator = InefficiencyCalculator(self.hurst_method, self.params)
        self.inefficiency_df = pd.DataFrame({col: self.inefficiency_calculator.calculate_inefficiency(self.data[col]) for col in self.data.columns})

    def _compute_conditional_correlations(self):
        """
        Call the DCC module and launch the dynamic conditional correlation.
        """
        # df_input = self.inefficiency_df.pct_change().dropna()
        df_input = self.inefficiency_df
        garch = self.garch_type.value()
        garch.fit(df_input)
        self.dcc = garch.get_cc_matrix()
        plt.figure(figsize=(10, 6))
        plt.plot(df_input.index, self.dcc, label='FTSE100 & FTSEMIB')
        plt.xlabel("Date")
        plt.ylabel("Conditional Correlation")
        plt.title("DCC(1,1) : Conditional Correlation between FTSE100 and FTSEMIB")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _compute_granger_causality(self):
        """
        Call the GrangerCausality module and launch the causality estimation throught VAR modeling and Granger causality test.
        """
        pass

    def run(self):
        """
        Run the full process for each start_date, end_date.
        """

        self._compute_inefficiency()
        self.inefficiency_df.to_excel('output/inefficiency.xlsx', index=True)
        self._compute_conditional_correlations()
        self._compute_granger_causality()

        res = Results(
            inefficiency_df=self.inefficiency_df,
            dcc=self.dcc,
            granger_tests=self.granger_tests
        )

        res.generate()

