from src.etude_stat.comparaison_estimateurs import EstimatorComparison
from src.enums import HurstMethodType

hurst_methods = {
    "SWV SD": HurstMethodType.SCALED_WINDOWED_VARIANCE,
    "SWV LD": HurstMethodType.SCALED_WINDOWED_VARIANCE,
    "SWV BD": HurstMethodType.SCALED_WINDOWED_VARIANCE,
    'SV': HurstMethodType.SCALED_VARIANCE,
    "AM": HurstMethodType.ABSOLUTE_MOMENTS,
    "DFA": HurstMethodType.DFA_ANALYSIS,
    "RS": HurstMethodType.RS_ANALYSIS,
    "M-RS": HurstMethodType.MODIFIED_RS_ANALYSIS,
}
methods_params = [
    {"method": "SD", "exclusions": True},
    {"method": "LD", "exclusions": True},
    {"method": "BD", "exclusions": True},
    {},
    {},
    {"minimal": 20, "method": "L2"},
    {'window_size': 10},
    {'window_size': 10}
]

etude = EstimatorComparison(
    hurst_methods=hurst_methods,
    true_hurst_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    nb_simulations=1000,
    simulation_length=[250, 500, 1000, 1500, 2000],
    methods_params=methods_params,
    json_path="output/etude_estimateurs/all_results_1000_jules_run.json"
)
# etude.study_estimators()
# etude.plot_hurst_boxplots(simulation_length=1000)
# etude.plot_hurst_heatmap('mse')
# etude.plot_metric_vs_hurst(1000, 'bias', False, ["RS", "M-RS"])
# etude.plot_metric_vs_length(0.5, 'variance', False, ["DFA"])
# etude.plot_execution_times(0.5)
etude.plot_qq_plot( 0.5, 1000)
print("Done")