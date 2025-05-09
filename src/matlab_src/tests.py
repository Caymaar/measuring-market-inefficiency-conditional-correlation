import matlab.engine
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from matlab_wrapper import MatlabEngineWrapper

def main():
    # Charger les données
    inefficiency_data = pd.read_excel('src/matlab_src/inefficiency.xlsx')

    # Initialiser le wrapper MATLAB
    matlab_wrapper = MatlabEngineWrapper('src/matlab_src/matlab_scripts')

    # Exécuter les tests
    data_diff, adf_result = matlab_wrapper.ensure_stationarity(inefficiency_data.iloc[:, 1:], lag=9, threshold=0.05)
    arch_df = matlab_wrapper.perform_arch_test(data_diff, 5)

    # Afficher les résultats
    print("=== Résultats ADF ===")
    print(adf_result) 
    print("\n=== Résultats ARCH ===")
    print(arch_df)

    # df_conds_vol, df_resids = matlab_wrapper.estimate_garch_volatility(data_diff)

    # # Sauvegarde ou affichage
    # print("\n=== Volatilité conditionnelle ===")
    # print(df_conds_vol.head())
    # print("\n=== Résidus standardisés ===")
    # print(df_resids.head())

    #cov_dcc, corr_dcc = matlab_wrapper.compute_all_dcc(data_diff)

    var_results, granger_results = matlab_wrapper.compute_all_var(data_diff)
    print("\n=== Résultats VAR ===")
    for key, result in var_results.items():
        print(f"\n=== VAR entre {key} ===")
        print(result)
    
    print("\n=== Résultats Granger ===")
    for key, result in granger_results.items():
        print(f"\n=== Granger entre {key} ===")
        print(result)

    # Fermer l'instance MATLAB
    matlab_wrapper.stop()

if __name__ == "__main__":
    main()
