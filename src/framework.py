from typing import Dict
import pandas as pd
from .enums import HurstMethodType, GarchMethodType
from .inefficiency_calculator import InefficiencyCalculator
from .results import Results
from .garch.dcc.dcc_implementation_package_cleaned import mgarch
from .matlab_src.matlab_wrapper import MatlabEngineWrapper
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
        self.matlab_wrapper = MatlabEngineWrapper('src/matlab_src/matlab_scripts')

        # Initialize the output series
        self.inefficiency_df = pd.DataFrame()
        self.corr_dcc = pd.DataFrame()
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

        # Compute ADF test and ARCH test
        data_diff, adf_result = self.matlab_wrapper.ensure_stationarity(self.inefficiency_df.dropna(), lag=9, threshold=0.05)
        arch_df = self.matlab_wrapper.perform_arch_test(data_diff,12)
        
        print("=== Résultats ADF ===")
        print(adf_result) 
        print("\n=== Résultats ARCH ===")
        print(arch_df)

        # self._compute_conditional_correlations()
        # self._compute_granger_causality()

        df_conds_vol, df_resids = self.matlab_wrapper.estimate_garch_volatility(data_diff)

        df_to_compare = data_diff.copy()

        self.cov_dcc, self.corr_dcc = self.matlab_wrapper.compute_all_dcc(df_to_compare)

        self.var_results, self.granger_tests = self.matlab_wrapper.compute_all_var(df_to_compare)
        print("\n=== Résultats VAR ===")
        for key, result in self.var_results.items():
            print(f"\n=== VAR entre {key} ===")
            print(result)
        
        print("\n=== Résultats Granger ===")
        for key, result in self.granger_tests.items():
            print(f"\n=== Granger entre {key} ===")
            print(result)

        res = Results(
            inefficiency_df=self.inefficiency_df,
            dcc=self.corr_dcc,
            var_results=self.var_results,
            granger_tests=self.granger_tests
        )

        res.generate()
        # Fermer l'instance MATLAB
        self.matlab_wrapper.stop()

