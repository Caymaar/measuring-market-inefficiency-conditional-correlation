import matlab.engine
import pandas as pd
import re


class MatlabEngineWrapper:
    def __init__(self, matlab_script_path):
        # Démarre une instance MATLAB
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(matlab_script_path, nargout=0)

    def stop(self):
        """Arrête l'instance MATLAB."""
        self.engine.quit()

    def perform_adf_test(self, data, lag):
        """Effectue le test ADF en appelant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        adf_stats, adf_pvalues = self.engine.performADFTest(data_matlab, lag, nargout=2)
        return self.format_results_df(data.columns.tolist(), adf_stats, adf_pvalues, test_name="ADF", nb_lags=lag)

    def perform_arch_test(self, data, lag):
        """Effectue le test ARCH en appelant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        arch_stats, arch_pvalues = self.engine.performARCHTest(data_matlab, lag, nargout=2)
        return self.format_results_df(data.columns.tolist(), arch_stats, arch_pvalues, test_name="ARCH", nb_lags=lag)
    
    
    def estimate_garch_volatility(self, data):
        """Appelle la fonction MATLAB pour estimer la volatilité conditionnelle et les résidus standardisés."""
        data_matlab = self.pandas_to_matlab_table(data)

        # Appelle la fonction MATLAB : elle doit renvoyer deux tables MATLAB
        cond_vols, resids = self.engine.estimateGARCH(data_matlab, nargout=2)

        df_conds_vol = pd.DataFrame(cond_vols, columns=data.columns.tolist())
        df_resids = pd.DataFrame(resids, columns=data.columns.tolist())
        return df_conds_vol, df_resids


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

def main():
    # Charger les données
    inefficiency_data = pd.read_excel('tests_matlab/inefficiency.xlsx')

    # Initialiser le wrapper MATLAB
    matlab_wrapper = MatlabEngineWrapper('c:/Users/bapdu/COMMUN/Dauphine/measuring-market-inefficiency-conditional-correlation/tests_matlab')

    # Exécuter les tests
    adf_df = matlab_wrapper.perform_adf_test(inefficiency_data.iloc[:, 1:], 9.0)
    arch_df = matlab_wrapper.perform_arch_test(inefficiency_data.iloc[:, 1:], 5.0)

    # Afficher les résultats
    print("=== Résultats ADF ===")
    print(adf_df)
    print("\n=== Résultats ARCH ===")
    print(arch_df)

    df_conds_vol, df_resids = matlab_wrapper.estimate_garch_volatility(inefficiency_data.iloc[:, 1:])

    # Sauvegarde ou affichage
    print("\n=== Volatilité conditionnelle ===")
    print(df_conds_vol.head())
    print("\n=== Résidus standardisés ===")
    print(df_resids.head())
    import matplotlib.pyplot as plt

    # Supposons que df_conds_vol et df_resids sont déjà calculés par matlab_wrapper
    # df_conds_vol et df_resids contiennent les données des volatilités et des résidus standardisés

    # Création de la figure et des sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot de la volatilité conditionnelle
    for column in df_conds_vol.columns:
        ax1.plot(df_conds_vol.index, df_conds_vol[column], label=column)
    ax1.set_title('Volatilité Conditionnelle')
    ax1.set_ylabel('Volatilité')
    ax1.legend()

    # Plot des résidus standardisés
    for column in df_resids.columns:
        ax2.plot(df_resids.index, df_resids[column], label=column)
    ax2.set_title('Résidus Standardisés')
    ax2.set_ylabel('Résidus')
    ax2.legend()

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()




    # Fermer l'instance MATLAB
    matlab_wrapper.stop()

if __name__ == "__main__":
    main()
