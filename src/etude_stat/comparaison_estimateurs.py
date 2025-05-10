from ..enums import HurstMethodType
from fbm import FBM
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import kstest
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import json
import math
import pandas as pd
import statsmodels.api as sm


class EstimatorComparison:
    """
    Class to compare different estimators of the Hurst exponent.
    """

    def __init__(self, 
                 hurst_methods: Dict[str, HurstMethodType] = HurstMethodType.SCALED_WINDOWED_VARIANCE, 
                 true_hurst_values: List[float] = [0.5], 
                 nb_simulations: int = 1000, 
                 simulation_length: List[float] = [1000],
                 methods_params: List[Dict] = None,
                 json_path: str = "output/etude_estimateurs/all_results.json"):
        """
        Initialize the EstimatorComparison with a list of Hurst methods.

        Parameters:
            hurst_methods (Dict[str, HurstMethodType]): Dict of Hurst estimation methods to compare
            true_hurst_values (List[float]): List of true Hurst exponent values for the simulations
            nb_simulations (int): Number of simulations to run for each method
            simulation_length (List[float]): Lengths of the synthetic time series to generate
            methods_params (List[Dict]): Dictionary of parameters for the methods
        """
        self.hurst_methods = hurst_methods
        self.true_hurst_values = true_hurst_values
        self.nb_simulations = nb_simulations
        self.simulation_length = simulation_length
        self.methods_params = methods_params if methods_params is not None else []
        self.json_path = json_path
        self.all_results = None

    def _generate_fbm(self, true_hurst: float, simulation_length: int) -> np.ndarray:
        """
        Generate a fractional Brownian motion time series.

        Parameters:
            true_hurst (float): The true Hurst exponent for the generated fBm
            simulation_length (int): The length of the synthetic time series to generate

        Returns:
            np.ndarray: A synthetic fBm time series
        """
        fbm = FBM(n=simulation_length, hurst=true_hurst, method="cholesky")

        return fbm.fbm()
    
    def _estimate_hursts(self, true_hurst: float, hurst_method: HurstMethodType, nb_simulations: int, simulation_length: int, method_params: Dict) -> Tuple[np.ndarray, float]:
        """
        Estimate n times the Hurst exponent using the specified method on the synthetic time series.

        Parameters:
            true_hurst (float): The true Hurst exponent for the generated synthetic fBm
            hurst_method (HurstMethodType): The method to use for estimating the Hurst exponent
            nb_simulations (int): The number of simulations to run
            simulation_length (int): The length of the synthetic time series to generate
            method_params (Dict): The parameters for the method

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - estimated_hurst (np.ndarray): The estimated Hurst values
                - time_taken (float): The mean time taken for the estimation
        """
        estimated_hurst = []
        estimation_times = []
        for _ in range(nb_simulations):
            synthetic_fbm = self._generate_fbm(true_hurst, simulation_length)
            hurst_estimator = hurst_method.value(synthetic_fbm, **method_params)
            start_time = time()
            estimated_hurst.append(hurst_estimator.estimate())
            estimation_times.append(time() - start_time)

        return np.array(estimated_hurst), np.mean(estimation_times)
    
    def _compare_estimators(self, true_hurst: float, simulation_length: int) -> Dict:
        """
        Compare the estimators for a given Hurst value and simulation length.

        Parameters:
            true_hurst (float): The true Hurst exponent for the generated synthetic fBm
            simulation_length (int): The length of the synthetic time series to generate
        
        Returns:
            Dict: A Dictionary with the Hurst methods as keys and as values a Dictionary containing the estimated Hurst values, bias, variance, MSE, and KS statistic and time taken for the estimation.
        """
        results = {}
        for i, (method_name, hurst_method) in enumerate(self.hurst_methods.items()):
            estimated_hursts, simulation_time = self._estimate_hursts(true_hurst, hurst_method, self.nb_simulations, simulation_length, self.methods_params[i])
            results[method_name] = {
                    'distrib': list(estimated_hursts),
                    'mean': float(np.mean(estimated_hursts)),
                    'bias': float(np.mean(estimated_hursts) - true_hurst),
                    'variance': float(np.var(estimated_hursts)),
                    'mse': float(np.mean((estimated_hursts - true_hurst)**2)),
                    'ks_stat': float(kstest(estimated_hursts, 'norm', args=(np.mean(estimated_hursts), np.std(estimated_hursts)))[0]),
                    'time': float(simulation_time),
                }
            
        return results
    
    def study_estimators(self):
        """
        Run the estimator comparison for all study cases, i.e., for all true Hurst values and simulation lengths.
        """
        all_results = {}
        for simulation_length in self.simulation_length:
            all_results[simulation_length] = {}
            for true_hurst in self.true_hurst_values:
                all_results[simulation_length][true_hurst] = self._compare_estimators(true_hurst, simulation_length)

        self.all_results = all_results
        with open("output/etude_estimateurs/all_results.json", "w") as f:
            json.dump(all_results, f, indent=4)
        
    def _load_results(self):
        """
        Load the results either from thefrom a JSON file.
        
        Parameters:
            file_path (str): Path to the JSON file containing the results
        """
        if not self.all_results:
            with open(self.json_path, "r") as f:
                raw = json.load(f)
            results = {}
            for sim_len_str, hurst_dict in raw.items():
                sim_len = int(sim_len_str)

                new_hurst = {}
                for hurst_str, methods in hurst_dict.items():
                    hurst_val = float(hurst_str)
                    new_hurst[hurst_val] = methods

                results[sim_len] = new_hurst

            return results
        else:
            return self.all_results
        
    def plot_hurst_boxplots(self, simulation_length: int):
        """
        Plot boxplots of Hurst estimates for each method, with true H values highlighted.
        
        Parameters:
            simulation_length (int): The simulation length to plot results for
        """
        results = self._load_results()

        results: dict = results[simulation_length]

        TRUE_H_COLOR = "#e41a1c"  # rouge vif
        BOX_ALPHA = 0.7

        for true_h, method_results in results.items():
            labels = list(method_results.keys())
            data   = [res['distrib'] for res in method_results.values()]

            fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5))

            bp = ax.boxplot(data,
                            vert=True,
                            patch_artist=True,
                            widths=0.6)

            for patch in bp['boxes']:
                patch.set_facecolor("#377eb8")
                patch.set_alpha(BOX_ALPHA)

            ax.axhline(true_h,
                    color=TRUE_H_COLOR,
                    linestyle='--',
                    linewidth=2,
                    alpha=0.9,
                    zorder=2)
            ax.axhspan(true_h - true_h * 0.05,
                    true_h + true_h * 0.05,
                    color=TRUE_H_COLOR,
                    alpha=0.1,
                    zorder=1)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            ax.set_ylabel("Estimation de H", fontsize=12)
            ax.set_title(f"Simulation Length = {simulation_length}  |  True H = {true_h:.2f}",
                        pad=12, fontsize=13)

            plt.tight_layout()
            fig.savefig(f"output/etude_estimateurs/BoxPlot_L{simulation_length}_H{str(true_h).replace('.', '_')}.png", dpi=150)
            plt.close(fig)

    def plot_hurst_heatmap(self, metric: str = 'mse'):
        """
        Generate heatmaps of estimator performance across true H values and simulation lengths.
        
        Parameters:
            metric (str): Metric to visualize ('mse', 'bias', or 'variance')
        """
        results = self._load_results()
        all_results: dict = results

        true_hursts = sorted(next(iter(all_results.values())).keys())
        sim_lengths = sorted(all_results.keys())
        methods = list(next(iter(next(iter(all_results.values())).values())).keys())
        
        n_methods = len(methods)
        n_cols = 3
        n_rows = math.ceil(n_methods / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
        axes = axes.flatten()  # To access axes[i] easily even in grid

        for idx, method in enumerate(methods):
            ax = axes[idx]
            data = np.zeros((len(true_hursts), len(sim_lengths)))
            for i, true_h in enumerate(true_hursts):
                for j, n in enumerate(sim_lengths):
                    data[i, j] = all_results[n][true_h][method][metric]

            sns.heatmap(
                data,
                ax=ax,
                annot=True,
                norm=LogNorm(),
                fmt=".3f",
                cmap="YlOrRd_r",
                cbar_kws={'label': f"{metric.capitalize()} (Log Scale)"},
                xticklabels=sim_lengths,
                yticklabels=[f"{h:.1f}" for h in true_hursts]
            )

            ax.set_title(f"{method}\n{metric.capitalize()}")
            ax.set_xlabel("Series Length")
            ax.set_ylabel("True Hurst")

        # Hide unused subplots
        for k in range(len(methods), len(axes)):
            fig.delaxes(axes[k])
        
        plt.tight_layout()
        fig.savefig(f"output/etude_estimateurs/{metric.capitalize()}HeatMap.png", dpi=150)
        plt.close(fig)

    def plot_metric_vs_hurst(self, simulation_length: int, metric: str = 'mse', show_confidence: bool = True, exclusions: list = None):
        """
        Plot performance metric vs true Hurst value for all methods.
        
        Parameters:
            simulation_length (int): Simulation length to analyze
            metric (str): Metric to plot ('mse', 'bias', 'variance')
            show_confidence (bool): Whether to display 95% confidence intervals
        """
        results = self._load_results()
        results = results[simulation_length]
        true_hursts = sorted(results.keys())
        methods = list(next(iter(results.values())).keys())
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']
        colors = plt.cm.tab10.colors
        
        for i, method in enumerate(methods):
            if exclusions and method in exclusions:
                continue
            metric_values = np.abs([results[h][method][metric] for h in true_hursts])
            
            if show_confidence and metric != 'variance':
                std_errors = [np.std(results[h][method]['distrib'])/np.sqrt(len(results[h][method]['distrib'])) for h in true_hursts]
                lower = [v - 1.96*se for v, se in zip(metric_values, std_errors)]
                upper = [v + 1.96*se for v, se in zip(metric_values, std_errors)]
            
            line = ax.plot(true_hursts, metric_values,
                   label=method,
                   marker=markers[i % len(markers)],
                   linestyle='-',
                   linewidth=2,
                   color=colors[i],
                   alpha=1.0)[0]
            
            if show_confidence and metric != 'variance':
                ax.fill_between(true_hursts, lower, upper,
                color=line.get_color(), alpha=0.1, zorder=1)
                ax.plot(true_hursts, lower, '--', color=line.get_color(), alpha=0.4)
                ax.plot(true_hursts, upper, '--', color=line.get_color(), alpha=0.4)

        ax.set_xlabel('True Hurst Exponent', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs True Hurst (n={simulation_length})', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        

        ax.set_xlim(min(true_hursts)-0.05, max(true_hursts)+0.05)
        if min(true_hursts) < 0.5 and max(true_hursts) > 0.5:
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"output/etude_estimateurs/{metric.capitalize()}vsHurst.png", bbox_inches='tight')
        plt.close()

    def plot_metric_vs_length(self, true_hurst: float, metric: str = 'mse', show_confidence: bool = True, exclusions: list = None):
        """
        Plot performance metric vs simulation length for all methods at a given true H.
        
        Parameters:
            true_hurst (float): True Hurst value to analyze
            metric (str): Metric to plot ('mse', 'bias', 'variance')
            show_confidence (bool): Whether to display 95% confidence intervals
        """
        results = self._load_results()
        lengths = sorted(results.keys())
        methods = list(results[lengths[0]][true_hurst].keys())

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']
        colors = plt.cm.tab10.colors
        
        for i, method in enumerate(methods):
            if exclusions and method in exclusions:
                continue
            metric_values = []
            std_errors = []
            
            for length in lengths:
                res = results[length][true_hurst][method]
                metric_values.append(np.abs(res[metric]))
                if show_confidence and metric != 'variance':
                    std_errors.append(np.std(res['distrib'])/np.sqrt(len(res['distrib'])))
            
            lengths_arr = np.array(lengths)
            line = ax.plot(lengths_arr, metric_values,
                        label=method,
                        marker=markers[i % len(markers)],
                        linestyle='-',
                        linewidth=2,
                        color=colors[i],
                        alpha=1.0)[0]
            
            if show_confidence and metric != 'variance' and std_errors:
                std_errors = np.array(std_errors)
                lower = metric_values - 1.96*std_errors
                upper = metric_values + 1.96*std_errors
                
                ax.fill_between(lengths_arr, lower, upper,
                            color=line.get_color(),
                            alpha=0.1,
                            zorder=1)
                ax.plot(lengths_arr, lower, '--',
                    color=line.get_color(),
                    alpha=0.4)
                ax.plot(lengths_arr, upper, '--',
                    color=line.get_color(),
                    alpha=0.4)

        # Customization matching your style
        ax.set_xlabel('Simulation Length', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Sample Size (True H = {true_hurst})', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"output/etude_estimateurs/{metric.capitalize()}vsLength.png", bbox_inches='tight')
        plt.close()

    def plot_execution_times(self, true_hurst: float):
        """
        Plot a bar chart comparing execution times (in milliseconds) across methods and series lengths,
        for a given true Hurst exponent. Uses 'hue' for simulation length.

        Parameters:
            true_hurst (float): The Hurst exponent to filter on
        """
        results = self._load_results()

        data = []
        for sim_length, hurst_dict in results.items():
            if true_hurst in hurst_dict:
                for method, metrics in hurst_dict[true_hurst].items():
                    data.append({
                        "Method": method,
                        "Execution Time (ms)": metrics['time'] * 1000,  # Convert to ms
                        "Series Length": sim_length
                    })

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="Method",
            y="Execution Time (ms)",
            hue="Series Length",
            palette="viridis"
        )

        plt.title(f"Average execution time per Method (True H = {true_hurst})")
        plt.ylabel("Execution Time (ms)")
        plt.xlabel("Method")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig(f"output/etude_estimateurs/ExecTime_BarPlot_H{str(true_hurst).replace('.', '_')}.png", dpi=150)
        plt.close()

    def plot_qq_plot(self, true_hurst: float, simulation_length: int):
        """
        Generate a grid of QQ-plots for all methods with normality statistics.

        Parameters:
            true_hurst (float): The true Hurst value used for simulation
            simulation_length (int): The simulation length used
        """
        results = self._load_results()
        
        try:
            method_results = results[simulation_length][true_hurst]
        except KeyError:
            print("Invalid Hurst value or simulation length.")
            return

        methods = list(method_results.keys())
        n_methods = len(methods)
        
        # Create subplot grid
        n_cols = 3
        n_rows = math.ceil(n_methods / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        axes = axes.flatten()

        for idx, method in enumerate(methods):
            ax = axes[idx]
            estimates = method_results[method]['distrib']
            ks_stat = method_results[method]['ks_stat']
            
            # Create QQ-plot
            sm.qqplot(np.array(estimates), line='s', ax=ax)
            
            # Add normality annotation
            ax.annotate(f"KS = {ks_stat:.3f}", 
                        xy=(0.05, 0.85), 
                        xycoords='axes fraction',
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            
            # Customize subplot
            ax.set_title(method, fontsize=10, pad=5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

        # Hide empty subplots
        for idx in range(len(methods), len(axes)):
            fig.delaxes(axes[idx])

        # Main title and layout
        plt.suptitle(f"QQ-plots (H={true_hurst}, L={simulation_length})", 
                    y=1.02, fontsize=12)
        plt.tight_layout()
        
        # Save and close
        plt.savefig(f"output/etude_estimateurs/QQGrid_H{str(true_hurst).replace('.', '_')}_L{simulation_length}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()