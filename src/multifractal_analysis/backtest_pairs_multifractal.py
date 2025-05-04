from mutlifractal_framework import MultifractalFramework
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_data, get_config


def compute_performance_stats(returns, freq=252):
    daily_returns = returns.dropna()
    if len(daily_returns) == 0:
        return np.nan, np.nan, np.nan, np.nan
    total_ret = (1 + daily_returns).prod() - 1
    nb_obs = len(daily_returns)
    annual_return = (1 + total_ret) ** (freq / nb_obs) - 1
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(freq)
    sharpe_ratio = annual_return / annual_vol
    cum_curve = (1 + daily_returns).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    return annual_return, annual_vol, sharpe_ratio, max_drawdown


if __name__ == "__main__":
    # Example usage
    # Liste de vos indices
    names = ["ftse100", "s&p500", "ftsemib", "ssec"]

    # Chemin de base vers les CSV
    DATA_DIR = (
        r"C:\Users\admin\Desktop\cours dauphine\S2\GQ2\measuring-market-inefficiency-conditional-correlation\data"
    )

    for ticker1, ticker2 in itertools.combinations(names, 2):
        # --- Chargement et calcul des rendements ---

        series1 = get_data(ticker1)
        series2 = get_data(ticker2)

        # Calcul des rendements
        ret1 = np.log(series1).diff().dropna()
        ret2 = np.log(series2).diff().dropna()

        # Alignez les deux séries
        data = pd.DataFrame({ticker1: ret1, ticker2: ret2}).dropna()

        # --- Multifractal framework & backtest ---
        mf = MultifractalFramework(
            data,
            ticker1=ticker1,
            ticker2=ticker2,
            window_hurst=120,
            window_mfdfa=1004,
            q_list=np.linspace(-3, 3, 13),
            scales=np.unique(np.logspace(np.log10(10), np.log10(500), 10, dtype=int)),
            order=1,
            backtest=True,
            verbose=True,
        )
        mf.compute_inefficiency_index()
        mf.compute_momentum()
        mf.compute_positions_with_inefficiency()
        mf.run_backtest()
        port_ret = mf.portfolio_returns.dropna()

        # Recalage des dates pour le plot
        common_index = port_ret.index.intersection(ret1.index).intersection(ret2.index)
        ret1 = ret1.loc[common_index]
        ret2 = ret2.loc[common_index]
        port_ret = port_ret.loc[common_index]
        start = mf.positions.index[0]
        cum1 = (1 + ret1.loc[start:]).cumprod()
        cum2 = (1 + ret2.loc[start:]).cumprod()
        cum_strat = (1 + port_ret).cumprod()
        porfolio_50_50_returns = 0.5 * ret1.loc[start:] + 0.5 * ret2.loc[start:]
        porfolio_50_50 = (1 + porfolio_50_50_returns).cumprod()

        # --- Affichage des rendements cumulés ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum1.index, y=cum1, mode="lines", name=f"{ticker1} Cumulative"))
        fig.add_trace(go.Scatter(x=cum2.index, y=cum2, mode="lines", name=f"{ticker2} Cumulative"))
        fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, mode="lines", name="Strategy Cumulative"))
        fig.add_trace(go.Scatter(x=porfolio_50_50.index, y=porfolio_50_50, mode="lines", name="50/50 Portfolio"))
        fig.update_layout(
            title=f"Cumulative Returns: {ticker1} vs {ticker2}", xaxis_title="Date", yaxis_title="Cumulative Returns"
        )
        fig.show()

        # --- Statistiques de performance ---
        stats_strat = compute_performance_stats(port_ret.loc[start:]) * 100
        stats1 = compute_performance_stats(ret1.loc[start:]) * 100
        stats2 = compute_performance_stats(ret2.loc[start:]) * 100
        stats_50_50 = compute_performance_stats(porfolio_50_50_returns.loc[start:]) * 100

        results = pd.DataFrame(
            {
                "Strategy": ["ModifOverlap120", f"Long Only {ticker1}", f"Long Only {ticker2}", "50/50 Portfolio"],
                "Annual Return": [stats_strat[0], stats1[0], stats2[0], stats_50_50[0]],
                "Annual Volatility": [stats_strat[1], stats1[1], stats2[1], stats_50_50[1]],
                "Sharpe Ratio": [stats_strat[2], stats1[2], stats2[2], stats_50_50[2]],
                "Max Drawdown": [stats_strat[3], stats1[3], stats2[3], stats_50_50[3]],
            }
        )

        # Affichage des résultats
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)

        out_fn = os.path.join(output_dir, f"performance_{ticker1}_{ticker2}.csv")
        results.to_csv(out_fn, index=False)
        print(f"\n=== Performance for {ticker1} vs {ticker2} ===")
        print(results)
