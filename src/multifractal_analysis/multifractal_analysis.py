from mutlifractal_framework import MultifractalFramework
import numpy as np
import plotly.graph_objects as go
import sys
import os

# on remonte d’un cran : src/multifractal_analysis → src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_data, get_config


if __name__ == "__main__":

    file_name = ["ftse100", "s&p500", "ftsemib", "ssec"]

    fig = go.Figure()
    for name in file_name:

        data = get_data(name)
        data = np.log(data).diff().dropna()
        mf = MultifractalFramework(
            data,
            ticker1="",
            ticker2="",
            window_hurst=0,
            window_mfdfa=1004,
            q_list=np.linspace(-3, 3, 13),
            scales=np.unique(np.logspace(np.log10(10), np.log10(1000), 10, dtype=int)),
            order=1,
            backtest=False,
            verbose=True,
        )
        mf.compute_multifractal()
        mf.plot_results(surogate=False, shuffle=True)
        fig.add_trace(go.Scatter(x=mf.alpha, y=mf.f_alpha, mode="lines+markers", name=name))
    fig.update_layout(title="Multifractal Spectrum ", xaxis_title="α", yaxis_title="f(α)", template="plotly_white")
    fig.show()
