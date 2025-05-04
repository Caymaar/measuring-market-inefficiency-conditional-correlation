from mutlifractal_framework import MultifractalFramework
import numpy as np
import plotly.graph_objects as go
import sys
import os
import pandas as pd

# on remonte d’un cran : src/multifractal_analysis → src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_data, get_config
from scipy.stats import kurtosis



if __name__ == "__main__":
    file_name = ["ftse100", "s&p500", "ftsemib", "ssec"]
    fig = go.Figure()
    np.random.seed(42)

    # Paramètres communs pour l'analyse multifractale
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10), np.log10(400), 10, dtype=int))

    # Boucle sur chaque indice
    for name in file_name:
        data = get_data(name)
        returns = np.log(data).diff().dropna()
        returns = returns.loc['1998-01-02':'2025-03-31']
        print(f"{name} → kurtosis: {kurtosis(returns):.4f}, length: {len(returns)}")

        mf = MultifractalFramework(
            returns,
            ticker1="",
            ticker2="",
            window_hurst=0,
            window_mfdfa=1004,
            q_list=q_list,
            scales=scales,
            order=1,
            backtest=False,
            verbose=True,
        )
        mf.compute_multifractal()
        mf.plot_results(shuffle=True, surogate=True)
        fig.add_trace(go.Scatter(
            x=mf.alpha,
            y=mf.f_alpha,
            mode="lines+markers",
            name=name
        ))

    # --- Spectre multifractal d'une marche aléatoire ---
    # Génération d'une marche aléatoire de même longueur que FTSE100
    # ref = get_data(file_name[0])
    # ref_returns = np.log(ref).diff().dropna().loc['1998-01-02':'2025-03-31']
    #
    # # Utilisation directe des retours aléatoires
    # random_returns = pd.Series(
    #     np.random.randn(len(ref_returns)),
    #     index=ref_returns.index,
    #     name='Random Returns'
    # )
    # print(f"Random Returns → kurtosis: {kurtosis(random_returns):.4f}, length: {len(random_returns)}")
    #
    # mf_rw = MultifractalFramework(
    #     random_returns,
    #     ticker1="",
    #     ticker2="",
    #     window_hurst=0,
    #     window_mfdfa=1004,
    #     q_list=q_list,
    #     scales=scales,
    #     order=1,
    #     backtest=False,
    #     verbose=False,
    # )
    # mf_rw.compute_multifractal()
    # fig.add_trace(go.Scatter(
    #     x=mf_rw.alpha,
    #     y=mf_rw.f_alpha,
    #     mode="lines+markers",
    #     name="Random Returns",
    # ))
    #
    # # Mise à jour du graphique
    # fig.update_layout(
    #     title="Multifractal Spectrum f(α) for Different Indices and Random Walk",
    #     xaxis_title="α",
    #     yaxis_title="f(α)",
    #     template="plotly_white"
    # )
    # fig.show()

