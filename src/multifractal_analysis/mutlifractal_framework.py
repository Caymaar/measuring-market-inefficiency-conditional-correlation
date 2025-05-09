import sys
import os
from typing import Optional, Sequence, Union, Tuple, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.multifractal_analysis.utils_multifractal import ComputeMFDFA
from src.utils import get_data, get_config
from src.hurst_estimation.modified_rescaled_range import ModifiedRSMethod
from src.hurst_estimation.abstract_estimation import AbstractHurstEstimator


class MultifractalFramework:
    """
    Cadre multifractal pour gérer l'analyse multifractale.
    """

    def __init__(
        self,
        data: pd.Series,
        window_hurst: int = 0,
        hurst_estimator: Type[AbstractHurstEstimator] = ModifiedRSMethod,
        window_mfdfa: int = 0,
        q_list: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None,
        order: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Initialise le cadre multifractal.

        Args:
            data (pd.Series) : Données d'entrée pour l'analyse multifractale.
            window_hurst (int) : Taille de la fenêtre pour le calcul de Hurst (rolling). Défaut 0.
            window_mfdfa (int) : Taille de la fenêtre pour le calcul MFDFA en rolling. Défaut 0.
            q_list (np.ndarray, optionnel) : Liste des exposants q. Par défaut np.linspace(-5, 5, 21).
            scales (np.ndarray, optionnel) : Échelles pour MFDFA. Par défaut logspace de 10 à 500.
            order (int) : Ordre du polynôme pour MFDFA. Défaut 1.
            backtest (bool) : Active le backtest si True. Défaut False.
            verbose (bool) : Active les messages détaillés. Défaut False.
        """

        self.data: pd.Series = data
        self.window_hurst: int = window_hurst
        self.window_mfdfa: int = window_mfdfa
        self.q_list: np.ndarray = q_list if q_list is not None else np.linspace(-3, 3, 14)
        self.scales: np.ndarray = (
            scales
            if scales is not None
            else np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
        )
        self.order: int = order
        self.verbose: bool = verbose

        # attributs
        self.alpha_widths: list = []
        self.delta_alpha_diff: Optional[pd.Series] = None
        self.inefficiency_index: Optional[pd.Series] = None
        self.rolling_hurst: Optional[pd.Series] = None
        self.hurst_estimator: Type[AbstractHurstEstimator] = hurst_estimator



    def compute_multifractal(self) -> None:
        """
        Calcule le spectre multifractal des données d'entrée.
        """
        if self.verbose:
            print("[Verbose] compute_multifractal début…")
        self.Fq = ComputeMFDFA.mfdfa(self.data, self.scales, self.q_list, self.order)
        self.h_q = ComputeMFDFA.compute_h_q(self.q_list, self.Fq, self.scales)
        self.alpha, self.f_alpha = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q)

        self.shuffle = ComputeMFDFA.shuffle(self.data)
        self.surogate = ComputeMFDFA.surrogate_gaussian_corr(self.data)

        self.Fq_shuffle = ComputeMFDFA.mfdfa(self.shuffle, self.scales, self.q_list, self.order)
        self.Fq_surogate = ComputeMFDFA.mfdfa(self.surogate, self.scales, self.q_list, self.order)

        self.h_q_shuffle = ComputeMFDFA.compute_h_q(self.q_list, self.Fq_shuffle, self.scales)
        self.h_q_surogate = ComputeMFDFA.compute_h_q(self.q_list, self.Fq_surogate, self.scales)

        self.alpha_shuffle, self.f_alpha_shuffle = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q_shuffle)
        self.alpha_surogate, self.f_alpha_surogate = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q_surogate)
        if self.verbose:
            print("[Verbose] compute_multifractal terminé")

    def compute_delta_alpha_diff(self) -> pd.Series:
        """
        Calcule la différence de largeur multifractale surrogate (∆α) rolling pour isoler la multifractalité lié
        au corrélation de long terme.

        Returns:
            pd.Series: ∆α surrogate
        """
        if self.verbose:
            print("[Verbose] compute_delta_alpha_diff…")
        if self.delta_alpha_diff is None:
            self.delta_alpha_diff = ComputeMFDFA.mfdfa_rolling(
                self.data, self.window_mfdfa, self.q_list, self.scales, self.order
            )

        if self.verbose:
            print("[Verbose] compute_delta_alpha_diff terminé")
        return self.delta_alpha_diff

    def compute_inefficiency_index(self) -> pd.Series:
        """
        Calcule l'indice d'inefficacité: ∆α × |H_rolling - 0.5|.

        Returns:
            pd.Series: inefficiency_index
        """
        if self.verbose:
            print("[Verbose] compute_inefficiency_index…")
        if self.delta_alpha_diff is None:
            self.compute_delta_alpha_diff()
        if self.rolling_hurst is None:
            self.rolling_hurst = (
                self.data
                .rolling(window=self.window_hurst)
                .apply(
                    lambda x: ModifiedRSMethod(
                        time_series=x.values,
                        window_size=len(x)
                    ).estimate(),
                    raw=False
                )
            )
        common_dates = self.delta_alpha_diff.index.intersection(self.rolling_hurst.index)
        self.delta_alpha_diff = self.delta_alpha_diff.loc[common_dates]
        self.rolling_hurst = self.rolling_hurst.loc[common_dates]

        self.inefficiency_index = self.delta_alpha_diff * (self.rolling_hurst - 0.5).abs()
        if self.verbose:
            print("[Verbose] compute_inefficiency_index terminé")
        return pd.Series(self.inefficiency_index, index=self.rolling_hurst.index, name="inefficiency_index")


    def plot_results(self, shuffle: bool = False, surogate: bool = False) -> None:
        """
        Trace h(q), spectres multifractaux avec options shuffle/surogate.
        """
        if self.verbose:
            print(f"[Verbose] plot_results (shuffle={shuffle}, surogate={surogate})…")
        # h(q)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.q_list, y=self.h_q, mode="lines+markers", name=f"h(q) {self.data.name}"))
        fig.update_layout(
            title=f"Spectre h(q) {self.data.name}", xaxis_title="q", yaxis_title="h(q)", template="plotly_white"
        )
        fig.show()
        # multifractal
        if shuffle and not surogate:
            # shuffle only
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig1.add_trace(go.Scatter(x=self.alpha_shuffle, y=self.f_alpha_shuffle, mode="lines+markers", name="f(α) shuffle"))
            fig1.update_layout(
                title="Spectre multifractal shuffle", xaxis_title="α", yaxis_title="f(α)", template="plotly_white"
            )
            fig1.show()
        elif surogate and not shuffle:
            # surogate only
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig2.add_trace(
                go.Scatter(x=self.alpha_surogate, y=self.f_alpha_surogate, mode="lines+markers", name="f(α) surogate")
            )
            fig2.update_layout(
                title="Spectre multifractal surogate", xaxis_title="α", yaxis_title="f(α)", template="plotly_white"
            )
            fig2.show()
        elif surogate and shuffle:
            # both
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig3.add_trace(go.Scatter(x=self.alpha_shuffle, y=self.f_alpha_shuffle, mode="lines+markers", name="f(α) shuffle"))
            fig3.add_trace(
                go.Scatter(x=self.alpha_surogate, y=self.f_alpha_surogate, mode="lines+markers", name="f(α) surogate")
            )
            fig3.update_layout(
                title="Spectre multifractal shuffle & surogate",
                xaxis_title="α",
                yaxis_title="f(α)",
                template="plotly_white",
            )
            fig3.show()
