import pandas as pd
from statsmodels.tsa.api import VAR


class GrangerCausalityTest:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame containing time series data.
        Each column represents a different inefficiency index or marker.
        """
        self.data = data.dropna()
        self.model = None
        self.lag_order = None

    def select_lags(self, maxlags: int = 10, criterion: str = 'aic'):
        """
        Select the optimal lag order using a specified criterion (e.g., AIC, BIC).
        """
        model = VAR(self.data)
        order_results = model.select_order(maxlags=maxlags)
        self.lag_order = getattr(order_results, criterion)
        print(f"Optimal lag order selected by {criterion.upper()}: {self.lag_order}")
        return self.lag_order

    def fit_var(self):
        """
        Fit the VAR model using the selected lag order.
        """
        if self.lag_order is None:
            self.select_lags()  # Default selection if not set
        model = VAR(self.data)
        self.model = model.fit(self.lag_order)
        print("VAR model fitted successfully.")
        return self.model

    def test_causality(self, target: str, causing: list, kind: str = 'f'):
        """
        Perform a Granger causality test to examine whether the variables in 'causing'
        Granger cause the variable 'target'.

        Parameters:
        - target: The dependent variable to be predicted.
        - causing: List of predictor variable names to test against 'target'.
        - kind: Type of test statistic (default 'f').

        Returns:
        - The causality test results.
        """
        if self.model is None:
            self.fit_var()
        test_result = self.model.test_causality(target, causing, kind=kind)
        return test_result