import pandas as pd
import matplotlib.pyplot as plt


class Results:
    """
    Stores the results of a Framework run i.e. the inefficiency series, DCC series, and Granger causality test results.
    Save the figures and the tables in the output directory.
    """

    def __init__(self, inefficiency_df: pd.DataFrame, dcc_df: pd.DataFrame, granger_tests: pd.DataFrame, plot: bool = True):
        """
        Parameters:
            inefficiency_series (pd.DataFrame): DataFrame containing the inefficiency series
            dcc_series (pd.DataFrame): DataFrame containing the DCC series
            granger_tests (pd.DataFrame): DataFrame containing the Granger causality test results
        """
        self.inefficiency_df = inefficiency_df
        self.dcc_df = dcc_df
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
        
        if self.dcc_df.empty:
            print("DCC series is empty. Skipping DCC plot.")
        else:
            # Plotting the dcc series
            num_series = self.dcc_df.shape[1]
            fig, axes = plt.subplots(num_series, 1, figsize=(10, 5 * num_series), sharex=True)

            for i, column in enumerate(self.dcc_df.columns):
                axes[i].plot(self.dcc_df.index, self.dcc_df[column], label=column, color='blue')
                axes[i].set_title(f'Dynamic Conditional Correlation : {column}', fontsize=14)
                axes[i].set_xlabel('Time', fontsize=12)
                axes[i].set_ylabel('DCC', fontsize=12)
                axes[i].grid(True, linestyle='--', alpha=0.6)
                axes[i].legend(fontsize=10)

            plt.tight_layout()
            plt.savefig('output/dcc_series_plot.png', dpi=300)
            if self.plot:
                plt.show()
            plt.close()

        if self.granger_tests == {}:
            print("Granger causality test results are empty. Skipping Granger causality plot.")
        else:
            # Saving the Granger causality test results
            self.granger_tests.to_csv('output/granger_tests_results.csv', index=False)
