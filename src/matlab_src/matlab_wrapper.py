import matlab.engine
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

class MatlabEngineWrapper:
    def __init__(self, matlab_script_path):
        # Démarre une instance MATLAB
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(matlab_script_path, nargout=0)
        self.engine.addpath(self.engine.genpath(f'{matlab_script_path}/mfe-toolbox-main'), nargout=0)
        print(self.engine.which('dcc', nargout=1))

    def stop(self):
        """Arrête l'instance MATLAB."""
        self.engine.quit()

    def perform_adf_test(self, data, lag):
        """Effectue le test ADF en appelant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        adf_stats, adf_pvalues = self.engine.performADFTest(data_matlab, float(lag), nargout=2)
        return self.format_results_df(data.columns.tolist(), adf_stats, adf_pvalues, test_name="ADF", nb_lags=lag)
    
    def ensure_stationarity(self, data, lag=9, threshold=0.05):
        """
        Applique le test ADF et différencie les séries non stationnaires.
        À la fin, toutes les séries ont la même taille (alignées sur la série la plus différenciée).
        """
        max_iter = 3
        transformed_data = data.copy()
        diff_count = {col: 0 for col in data.columns}

        for _ in range(max_iter):
            adf_result = self.perform_adf_test(transformed_data, lag)
            non_stationary_cols = adf_result[adf_result["ADF_pValue"] > threshold]["MarketIndex"]

            if non_stationary_cols.empty:
                break

            for col in non_stationary_cols:
                transformed_data[col] = transformed_data[col].diff()
                diff_count[col] += 1

            transformed_data = transformed_data.dropna()
        adf_result['DiffCount'] = adf_result['MarketIndex'].map(diff_count)
        return transformed_data.reset_index(drop=True), adf_result


    def perform_arch_test(self, data, lag = 5):
        """Effectue le test ARCH en appelant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        arch_stats, arch_pvalues = self.engine.performARCHTest(data_matlab, float(lag), nargout=2)
        return self.format_results_df(data.columns.tolist(), arch_stats, arch_pvalues, test_name="ARCH", nb_lags=lag)
    
    
    def estimate_garch_volatility(self, data, max_p = 10, max_q = 10):
        """Appelle la fonction MATLAB pour estimer la volatilité conditionnelle et les résidus standardisés."""
        data_matlab = self.pandas_to_matlab_table(data)

        # Appelle la fonction MATLAB : elle doit renvoyer deux tables MATLAB
        cond_vols, resids, aic, pq = self.engine.estimateGARCH(data_matlab, float(max_p), float(max_q), nargout=4)

        df_conds_vol = pd.DataFrame(cond_vols, columns=data.columns.tolist())
        df_resids = pd.DataFrame(resids, columns=data.columns.tolist())

        return df_conds_vol, df_resids

    def compute_all_dcc(self, data):
        """Appelle la fonction MATLAB pour estimer la DCC entre toutes les colonnes."""

        cov_results = {}
        corr_results = {}

        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                data1 = data.iloc[:, i]
                data2 = data.iloc[:, j]
                cov_dcc, corr_dcc = self.compute_dcc(data1, data2)
                cov_results[f'{data.columns[i]}_{data.columns[j]}'] = cov_dcc
                corr_results[f'{data.columns[i]}_{data.columns[j]}'] = corr_dcc
   
        return pd.DataFrame(cov_results), pd.DataFrame(corr_results)
    
    def compute_dcc(self, data1, data2):
        """Appelle la fonction MATLAB pour estimer la DCC."""
        M, L, N, P, O, Q = 1, 0, 1, 1, 0, 1 # Paramètres par défaut
      
        # Convertir en format MATLAB
        data1_matlab = matlab.double(data1.values.tolist())  # 1 x T
        data2_matlab = matlab.double(data2.values.tolist())  # 1 x T
       
        cov_dcc, corr_dcc = self.engine.computeDCC(
            data1_matlab,   # T x 1
            data2_matlab,   # T x 1
            float(M),
            float(L),
            float(N),
            float(P),
            float(O),
            float(Q),
            nargout=2
        )
        cov_dcc_list = [x[0] for x in cov_dcc]
        corr_dcc_list = [x[0] for x in corr_dcc]
        return cov_dcc_list, corr_dcc_list

    def compute_all_var(self, data):
        """Appelle la fonction MATLAB pour estimer le VAR entre toutes les colonnes."""
        var_results = {}
        granger_results = {}
        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                data1 = data.iloc[:, i]
                data2 = data.iloc[:, j]
                result_table, granger_table = self.compute_var(data1, data2)
                var_results[f'{data.columns[i]}_{data.columns[j]}'] = result_table
                granger_results[f'{data.columns[i]}_{data.columns[j]}'] = granger_table

        return var_results, granger_results

    def compute_var(self, data1, data2):
        """Appelle la fonction MATLAB pour estimer un modèle VAR entre deux séries temporelles."""
        p = 1  # Ordre du modèle VAR (1)

        # Convertir les données pandas en format compatible avec MATLAB
        data1_matlab = matlab.double(data1.values.tolist())  # 1 x T
        data2_matlab = matlab.double(data2.values.tolist())  # 1 x T

        # Appeler la fonction MATLAB estimate_VAR
        result_table, granger_results = self.engine.estimate_VAR(data1_matlab, data2_matlab, float(p), nargout=2)

        # Convertir les résultats de MATLAB en un tableau numpy
        result_array = np.array(self.engine.table2array(result_table))

        # Créer des DataFrames pandas pour les résultats VAR et Granger
        result_df = pd.DataFrame(result_array, columns=['Estimate1', 'Estimate2', 'StdError', 'tValue1', 'tValue2', 'pValue1', 'pValue2'])

        # Format des résultats
        formatted_var_df = self.format_var_output(result_df, [data1.name, data2.name])
        formatted_granger_df = self.format_granger_output(granger_results, [data1.name, data2.name])

        return formatted_var_df, formatted_granger_df


    def pandas_to_matlab_table(self, df):
        """Convertit un DataFrame pandas en table MATLAB avec noms de colonnes valides (colonne = vecteur vertical)."""
        original_names = df.columns.tolist()
        sanitized_names = self.sanitize_matlab_names(original_names)

        # Assurer que chaque colonne est convertie en vecteur colonne (Nx1)
        for col, safe_col in zip(original_names, sanitized_names):
            # .tolist() donne une liste à 1D, on convertit en liste de listes pour obtenir un vecteur colonne
            column_data = [[v] for v in df[col].dropna().values.tolist()]
            self.engine.workspace[safe_col] = matlab.double(column_data)

        # Créer une table MATLAB avec les colonnes converties
        table_command = "T = table(" + ", ".join(sanitized_names) + ");"
        self.engine.eval(table_command, nargout=0)

        # Restaurer les noms d’origine dans la table MATLAB
        for i, orig_name in enumerate(original_names):
            self.engine.eval(f"T.Properties.VariableNames{{{i+1}}} = '{orig_name}';", nargout=0)

        return self.engine.workspace['T']

    def sanitize_matlab_names(self, names):
        """Nettoie les noms pour qu’ils soient valides en MATLAB."""
        sanitized = []
        for name in names:
            # Remplacer les caractères non alphanumériques par des underscores
            name_clean = re.sub(r'\W|^(?=\d)', '_', name)
            sanitized.append(name_clean)
        return sanitized
    
    def format_results_df(self, var_names, stats, pvalues, test_name="Test", nb_lags = 0):
        """Retourne un DataFrame formaté avec les résultats."""
        return pd.DataFrame({
            'MarketIndex': var_names,
            f'{test_name}_Statistic': [float(s) for s in stats[0]],
            f'{test_name}_pValue': [float(p) for p in pvalues[0]],
            "Lags": [nb_lags] * len(var_names),
        })
    
    def format_var_output(self, df, var_names):
        """
        Reformate un DataFrame de résultats VAR en structure longue.

        Parameters:
        - df : DataFrame avec colonnes Estimate1, Estimate2, StdError, tValue1, tValue2, pValue1, pValue2
        - var_names : liste avec les noms des deux variables (ex: ['FTSEMIB', 'FTSE100'])

        Returns:
        - DataFrame formaté avec une ligne par équation / variable
        """

        formatted = pd.DataFrame({
            'Equation': [f'{var_names[0]}', f'{var_names[0]}', f'{var_names[1]}', f'{var_names[1]}'],
            'Variable': [f'I({var_names[0]})t−1', f'I({var_names[1]})t−1',
                     f'I({var_names[0]})t−1', f'I({var_names[1]})t−1'],
            'Estimate': [df['Estimate1'][0], df['Estimate2'][0], df['Estimate1'][1], df['Estimate2'][1]],
            'StdError': [df['StdError'][0]] * 2 + [df['StdError'][1]] * 2,
            'tValue': [df['tValue1'][0], df['tValue2'][0], df['tValue1'][1], df['tValue2'][1]],
            'pValue': [df['pValue1'][0], df['pValue2'][0], df['pValue1'][1], df['pValue2'][1]]
        })
        return formatted

    import pandas as pd

    def format_granger_output(self, granger_results, names):
        """
        Format the Granger causality test results into a pandas DataFrame.
        
        Parameters:
        - granger_results: MATLAB object containing the Granger test results
        
        Returns:
        - DataFrame with columns: 'Test', 'WaldStat', 'pValue', 'Decision'
        """
        # Accessing the necessary fields from the granger_results MATLAB object
        test_field = self.engine.getfield(granger_results, 'Test')
        wald_stat_field = self.engine.getfield(granger_results, 'WaldStat')
        p_value_field = self.engine.getfield(granger_results, 'pValue')
        decision_field = self.engine.getfield(granger_results, 'Decision')
        
        # Convert each field to a list or array (depending on the type of the field)
        test_list = list(test_field)  # or use np.array(test_field) if needed
        wald_stat_list = [x[0] for x in  wald_stat_field]  # or np.array(wald_stat_field)
        p_value_list =  [x[0] for x in  p_value_field]  # or np.array(p_value_field)
        decision_list = list(decision_field)  # or np.array(decision_field)
        
        # Creating a DataFrame from the extracted fields
        granger_df = pd.DataFrame({
            'Y1' : names[0],
            'Y2' : names[1],
            'Test': test_list,
            'WaldStat': wald_stat_list,
            'pValue': p_value_list,
            'Decision': decision_list
        })
        
        return granger_df
