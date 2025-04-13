import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enums import HurstMethodType

class InefficiencyCalculator:
    def __init__(self, hurst_method_type: HurstMethodType, datas: dict, params: dict):

        self.hurst_method_type = hurst_method_type.value

    def _rolling_hurst(self, time_series: pd.Series) -> pd.Series:

        hurst_values = []
        index_list = []
        
        for i in range(len(time_series) - self.window + 1):
            window = time_series.iloc[i:i+self.window].values

            hurst_method = self.hurst_method_type(window, *self.params['method_params'])
            h = hurst_method.estimate()

            hurst_values.append(h)
            index_list.append(time_series.index[i+self.window - 1])    

        return pd.Series(hurst_values, index=index_list)      

    def calculate_inefficiency(self, prices: pd.Series) -> pd.Series:
        hurst_series = self._rolling_hurst(prices)
        return 0.5 - hurst_series

class InefficiencyVisualizer:
    def __init__(self, hurst_calculator, datas: dict, dates: dict):
        """
        Paramètres:
            hurst_calculator: instance de HurstCalculator
            datas: dict, clé = nom de la série, valeur = pd.Series de prix indexée par date
            dates: dict, clé = date de début (string ou datetime), valeur = date de fin (string ou datetime)
        """
        self.hurst_calculator = hurst_calculator
        self.datas = datas
        self.dates = dates
        
    def plot_inefficiencies(self):
        """
        Pour chaque période définie dans self.dates, filtre chaque série de prix,
        calcule l'inefficience via HurstCalculator et affiche les résultats dans un subplot
        disposé sur deux colonnes.
        """
        # Pour chaque période à explorer
        for start_date, end_date in self.dates.items():
            # Filtrer chaque série sur la période et compter le nombre de séries
            filtered_datas = {}
            for label, series in self.datas.items():
                # Calculate the adjusted start date by subtracting window size in business days
                adjusted_start = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(self.hurst_calculator.window)
                filtered_series = series.loc[adjusted_start:end_date]
                # Vérifier que la série n'est pas vide
                if not filtered_series.empty:
                    filtered_datas[label] = filtered_series
            n_series = len(filtered_datas)
            if n_series == 0:
                print(f"Aucune donnée pour la période {start_date} à {end_date}.")
                continue

            # Calcul du nombre de lignes pour 2 colonnes
            n_cols = 2
            n_rows = math.ceil(n_series / n_cols)
            
            # Création de la figure et des axes
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4), squeeze=False)
            axes = axes.flatten()
            
            # Pour chaque série filtrée, calcul et tracé de l'inefficience
            for idx, (label, series) in enumerate(filtered_datas.items()):
                ineff_series, _ = self.hurst_calculator.calculate_inefficiency(series)
                ax = axes[idx]
                ax.plot(ineff_series.index, ineff_series.values, label=label)
                ax.set_title(f"{label}\n({start_date} à {end_date})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Inefficience")
                ax.legend()
            
            # Supprimer les axes vides si nécessaire
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])
            
            fig.suptitle(f"Inefficience de marché du {start_date} au {end_date}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
