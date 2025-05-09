from mutlifractal_framework import MultifractalFramework
import os
import sys
import pandas as pd
import numpy as np

# on remonte d’un cran : src/multifractal_analysis → src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import get_data, get_config
from scipy.stats import kurtosis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.multifractal_analysis.utils_multifractal import ComputeMFDFA
from scipy.stats import jarque_bera

if __name__ == '__main__':

    # Liste de vos indices
    names = ["ftse100", "s&p500", "ftsemib", "ssec"]

    # Chemin de base vers les CSV
    DATA_DIR = (
        r"C:\Users\admin\Desktop\cours dauphine\S2\GQ2\measuring-market-inefficiency-conditional-correlation\data"
    )

    for ticker in names:
        # --- Chargement et calcul des rendements ---
        series1 = get_data(ticker)

        returns = np.log(series1).diff().dropna()

        returns = returns.loc['1998-01-02':'2025-03-31']
        print(f"{ticker} → kurtosis: {kurtosis(returns):.4f}, length: {len(returns)}")

        np.random.seed(42)
        surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
        surrogate_returns = pd.Series(surrogate_returns, index=returns.index)
        surrogate_returns.name = ticker

        stat, p_value = jarque_bera(surrogate_returns)
        print(f"Jarque-Bera test for {ticker}: stat={stat}, p-value={p_value}")

        i = 1

        while p_value < 0.05:
            np.random.seed(i + 42)
            surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
            surrogate_returns = pd.Series(surrogate_returns, index=returns.index)
            surrogate_returns.name = ticker
            stat, p_value = jarque_bera(surrogate_returns)
            print(f"Jarque-Bera test for {ticker}: stat={stat}, p-value={p_value}")
            i += 1


        # --- Multifractal framework & backtest ---
        mf = MultifractalFramework(
            surrogate_returns,
            window_hurst=120,
            window_mfdfa=252,
            q_list=np.linspace(-3, 3, 13),
            scales=np.unique(np.floor(np.logspace(np.log10(10), np.log10(50), 10)).astype(int)),
            order=1,
            verbose=True,
        )

        inef_index = mf.compute_inefficiency_index()

        common_dates = series1.index.intersection(inef_index.index)
        series1 = series1.loc[common_dates]
        inef_index = inef_index.loc[common_dates]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Log Prices", "Inefficiency Index"),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
        )
        fig.add_trace(
            go.Scatter(x=series1.index, y=np.log(series1), mode='lines', name=ticker),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=inef_index.index, y=inef_index, mode='lines', name="Inefficiency Index"),
            row=2, col=1
        )
        fig.update_layout(title_text=f"{ticker} - Inefficiency Index")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Log Price", row=1, col=1)
        fig.update_yaxes(title_text="Inefficiency Index", row=2, col=1)
        fig.show()


