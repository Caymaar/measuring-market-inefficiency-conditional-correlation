import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import math


class Results:
    """
    Stores the results of a Framework run i.e. the inefficiency series, DCC series, and Granger causality test results.
    Save the figures and the tables in the output directory.
    """

    def __init__(self, inefficiency_df: pd.DataFrame, dcc: np.ndarray, granger_tests: pd.DataFrame, plot: bool = True):
        """
        Parameters:
            inefficiency_series (pd.DataFrame): DataFrame containing the inefficiency series
            dcc_series (pd.DataFrame): DataFrame containing the DCC series
            granger_tests (pd.DataFrame): DataFrame containing the Granger causality test results
        """
        self.inefficiency_df = inefficiency_df
        self.dcc = dcc
        self.granger_tests = granger_tests
        
        self.plot = plot

    def generate(self):
        """
        Generate the results by saving the figures and the tables.
        """

        # Plotting the inefficiency series with two columns layout
        num_series = self.inefficiency_df.shape[1]
        num_rows = (num_series + 1) // 2  # Calculate the number of rows needed for two columns
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), sharex=True)
        axes = axes.flatten()  # Flatten the axes array for easier indexing

        for i, column in enumerate(self.inefficiency_df.columns):
            axes[i].plot(self.inefficiency_df.index, self.inefficiency_df[column], label=column, color='blue')
            axes[i].set_title(f'Inefficiency Series: {column}', fontsize=14)
            axes[i].set_xlabel('Time', fontsize=12)
            axes[i].set_ylabel('Inefficiency', fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].legend(fontsize=10)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('output/inefficiency_series_plot.png', dpi=300)
        if self.plot:
            plt.show()
        plt.close()
        
        if len(self.dcc) == 0:
            print("DCC series is empty. Skipping DCC plot.")
        else:
            # T, N, _ = self.dcc.shape
            # paires = list(combinations(range(N), 2))
            # M = len(paires)

            # # Définir la grille (nrows x ncols) la plus carrée possible
            # ncols = math.ceil(math.sqrt(M))
            # nrows = math.ceil(M / ncols)

            # fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
            # axes = axs.ravel()

            # for idx, (i, j) in enumerate(paires):
            #     ax = axes[idx]
            #     corr_series = self.dcc[:, i, j]
            #     ax.plot(corr_series)
            #     ax.set_title(f"Corrélation {i+1}-{j+1}")
            #     ax.set_xlabel("Date")
            #     ax.set_ylabel("ρₜ")
            
            # # Retirer les axes inutilisés
            # for k in range(M, len(axes)):
            #     fig.delaxes(axes[k])

            # # Ajustement et légende si besoin (ex. légende globale)
            # plt.tight_layout()
            # # fig.legend([...], bbox_to_anchor=(1.05,1), loc='upper left')
            # plt.show()
            num_series = self.dcc.shape[1]
            num_rows = (num_series + 1) // 2  # Calculate the number of rows needed for two columns
            fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), sharex=True)
            axes = axes.flatten()  # Flatten the axes array for easier indexing

            for i, column in enumerate(self.dcc.columns):
                axes[i].plot(self.dcc.index, self.dcc[column], label=column, color='blue')
                axes[i].set_title(f'DCC Correlation Series: {column}', fontsize=14)
                axes[i].set_xlabel('Time', fontsize=12)
                axes[i].set_ylabel('Correlation', fontsize=12)
                axes[i].grid(True, linestyle='--', alpha=0.6)
                axes[i].legend(fontsize=10)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig('output/correlation_series_plot.png', dpi=300)
            if self.plot:
                plt.show()
            plt.close()

        if self.granger_tests == {}:
            print("Granger causality test results are empty. Skipping Granger causality plot.")
        else:
            # Saving the Granger causality test results
            self.granger_tests.to_csv('output/granger_tests_results.csv', index=False)
