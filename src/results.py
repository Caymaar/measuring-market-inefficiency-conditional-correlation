import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import math
import matplotlib.dates as mdates
import seaborn as sns

sns.set(style="whitegrid", palette="muted")

class Results:
    """
    Stores the results of a Framework run i.e. the inefficiency series, DCC series, and Granger causality test results.
    Save the figures and the tables in the output directory.
    """

    def __init__(self, inefficiency_df: pd.DataFrame, dcc: np.ndarray, var_results : dict[pd.DataFrame], granger_tests: dict[pd.DataFrame], 
                 plot: bool = True, path: str = ''):
        """
        Parameters:
            inefficiency_series (pd.DataFrame): DataFrame containing the inefficiency series
            dcc_series (pd.DataFrame): DataFrame containing the DCC series
            granger_tests (pd.DataFrame): DataFrame containing the Granger causality test results
        """
        self.inefficiency_df = inefficiency_df
        self.dcc = dcc
        self.var_results = var_results
        self.granger_tests = granger_tests

        self.plot = plot
        self.path = path
        print("Results initialized with path: ", self.path)

    def generate(self):
        # Plotting the inefficiency series with two columns layout
        num_series = self.inefficiency_df.shape[1]
        num_rows = (num_series + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), sharex=True)
        axes = axes.flatten()
        self.inefficiency_df.index = pd.to_datetime(self.inefficiency_df.index)
        
        colors = sns.color_palette("tab10", n_colors=num_series)
        
        for i, column in enumerate(self.inefficiency_df.columns):
            axes[i].plot(self.inefficiency_df.index, self.inefficiency_df[column], label=column, color=colors[i])
            axes[i].set_title(f'Inefficiency Series: {column}', fontsize=14, weight='bold', color='darkblue', backgroundcolor='lightgray')
            axes[i].set_ylabel('Inefficiency', fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].legend(fontsize=10)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
            
            for label in axes[i].get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
                label.set_fontsize(10)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        #plt.savefig(self.path+'_inneficiency_series.png', dpi=300)
        if self.plot:
            plt.show()
        plt.close()

        if len(self.dcc) == 0:
            print("DCC series is empty. Skipping DCC plot.")
        else:
            num_series = self.dcc.shape[1]
            num_rows = (num_series + 2) // 3  # Now dividing by 3 instead of 2 for 3 columns
            fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), sharex=True)
            axes = axes.flatten()
            
            diff_dates = len(self.inefficiency_df.index) - len(self.dcc.index)
            self.dcc.index = pd.to_datetime(self.dcc.index)
            
            colors = sns.color_palette("tab10", n_colors=num_series)

            for i, column in enumerate(self.dcc.columns):
                axes[i].plot(self.inefficiency_df.index[diff_dates:], self.dcc[column], label=column, color=colors[i])
                axes[i].set_title(f'DCC Correlation Series: {column}', fontsize=14, weight='bold', color='darkgreen', backgroundcolor='lightyellow')
                axes[i].set_ylabel('Correlation', fontsize=12)
                axes[i].grid(True, linestyle='--', alpha=0.6)
                axes[i].legend(fontsize=10)
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
                
                for label in axes[i].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                    label.set_fontsize(10)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            #plt.savefig(self.path+'_dcc.png', dpi=300)
            if self.plot:
                plt.show()
            plt.close()

        if self.var_results == {}:
            print("VAR results are empty. Skipping Granger causality plot.")
        else:
            # Saving the Granger causality test results
            print("VAR results saving:")
            with pd.ExcelWriter(self.path+'_var_results.xlsx', engine='xlsxwriter') as writer:
                for sheet_name, df in self.var_results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        if self.granger_tests == {}:
            print("Granger causality test results are empty. Skipping Granger causality plot.")
        else:
            # Saving the Granger causality test results
            print("Granger causality test results saving:")
            with pd.ExcelWriter(self.path+'_granger_tests_results.xlsx', engine='xlsxwriter') as writer:
                for sheet_name, df in self.granger_tests.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
