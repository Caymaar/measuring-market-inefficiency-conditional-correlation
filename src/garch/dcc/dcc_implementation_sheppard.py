import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.linalg import sqrtm, inv, eigvals
import matplotlib.pyplot as plt
from src.hurst_calculation import HurstCalculator
import pandas as pd
from src.utils import get_data
from arch import arch_model

# ---------------------- Copy du code Matlab de Kevin sheppard le GOAT ----------------------

# --- Helper functions for converting between correlation matrix representations --- #
def corr_vech(R):
    """
    Return the lower-triangular (including diagonal) vectorized elements of a symmetric matrix R.
    """
    k = R.shape[0]
    return R[np.tril_indices(k)]


def vech(X):
    """
    Return the lower-triangular (including diagonal) elements of matrix X.
    """
    k = X.shape[0]
    return X[np.tril_indices(k)]


def r2z(R):
    """
    Convert off-diagonal elements of correlation matrix R to Fisher z-values.
    (Diagonal elements remain at 0.)
    """
    k = R.shape[0]
    z = []
    for i in range(1, k):
        for j in range(i):
            # Fisher z transformation
            z.append(0.5 * np.log((1 + R[i, j]) / (1 - R[i, j])))
    return np.array(z)


def z2r(z):
    """
    Reconstruct a correlation matrix from a vector of Fisher z-values.
    (Assumes the lower triangular part of a symmetric matrix.)
    """
    import math
    k = int((1 + math.sqrt(1 + 8 * len(z))) / 2)
    R = np.eye(k)
    idx = 0
    for i in range(1, k):
        for j in range(i):
            val = np.tanh(z[idx])
            R[i, j] = val
            R[j, i] = val
            idx += 1
    return R


# --- Stub functions that need to be filled with full implementations ---
def dcc_fit_variance(data2d, p, o, q, gjrType, tarchStartingVals):
    """
    For each asset (each column in data2d), fit a GARCH(1,1) model using the arch package.
    Returns:
      H: an array of shape (T, k) with the estimated conditional variances.
      univariate: a list of dictionaries for each asset containing estimated parameters.
    """
    T, k = data2d.shape
    H = np.zeros((T, k))
    univariate = []
    for i in range(k):
        series = data2d[:, i]
        # Fit a GARCH(1,1) model with zero mean
        am = arch_model(series, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        H[:, i] = res.conditional_volatility ** 2  # conditional variance at each time
        univ = {'parameters': res.params,
                'p': p[i] if np.isscalar(p) is False else p,
                'o': o[i] if np.isscalar(o) is False else o,
                'q': q[i] if np.isscalar(q) is False else q,
                'A': np.eye(len(res.params)),
                'scores': np.zeros((T, len(res.params)))}  # (dummy scores)
        univariate.append(univ)
    return H, univariate


# ---------------------------
# Real DCC Likelihood function (DCC(1,1), Gaussian innovations)
# ---------------------------
def dcc_likelihood(params, data2d, H):
    """
    Computes the negative log-likelihood for the DCC(1,1) model.

    INPUTS:
      params: array-like with two elements [a, b] from DCC dynamic equation.
      data2d: (T x k) array of zero-mean residuals (inefficiency indices).
      H:      (T x k) array of conditional variances estimated from the univariate GARCH models.

    Steps:
      1. Compute standardized residuals: z_t = r_t / sqrt(H[t, :])
      2. Estimate Q_bar = sample covariance matrix of z.
      3. Initialize Q_0 = Q_bar, then recursively update
             Q_t = (1 - a - b) * Q_bar + a * (z_{t-1} z_{t-1}ᵀ) + b * Q_{t-1}
         for t=1,...,T-1.
      4. Compute R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2).
      5. For t=1,...,T-1, accumulate the log-likelihood contribution:
             L_t = -0.5 * ( ln|R_t| + z_t' R_t^{-1} z_t ).
      6. Return the negative total log-likelihood.
    """
    T, k = data2d.shape
    # Compute standardized residuals z
    z = np.zeros((T, k))
    for t in range(T):
        # Avoid division by zero using np.sqrt(H[t, :]) > 0 (assumed positive)
        z[t, :] = data2d[t, :] / np.sqrt(H[t, :])
    Q_bar = np.cov(z, rowvar=False)
    Q = np.zeros((T, k, k))
    R_t = np.zeros((T, k, k))
    L = 0.0
    Q[0] = Q_bar.copy()
    for t in range(1, T):
        z_prev = z[t - 1, :].reshape(k, 1)
        Q[t] = (1 - params[0] - params[1]) * Q_bar + params[0] * (z_prev @ z_prev.T) + params[1] * Q[t - 1]
        diag_q = np.sqrt(np.diag(Q[t]))
        inv_diag = np.diag(1.0 / diag_q)
        R_t[t] = inv_diag @ Q[t] @ inv_diag
        sign, logdet = np.linalg.slogdet(R_t[t])
        if sign <= 0:
            return 1e6  # Penalty for non-positive definite correlation matrix
        term = z[t, :] @ np.linalg.solve(R_t[t], z[t, :])
        L += -0.5 * (logdet + term)
    return -L  # We minimize the negative log-likelihood


def gradient_2sided(f, x, *args, **kwargs):
    """
    Computes the two-sided finite-difference gradient of the function f at x.

    Parameters
    ----------
    f : callable
        Function handle such that f(x, *args, **kwargs) returns either a scalar
        or a tuple (fval, score) where score is a vector of individual scores.
    x : array_like, shape (M,)
        Parameter vector.
    *args, **kwargs :
        Additional arguments to pass to f.

    Returns
    -------
    G : ndarray, shape (M,)
        Finite-difference central gradient: G[i] = (f(x+e_i) - f(x-e_i)) / (2*h[i])
    Gt : ndarray, shape (T, M), optional
        Matrix of individual scores (if f returns a tuple (fval, score) when called).
        Each column j is given by (score_forward[:,j] - score_backward[:,j])/(2*h[j]).
        Returned only if f returns two outputs.

    Example
    -------
    >>> # Define a function that returns both a function value and its individual score vector.
    >>> def test_func(x):
    ...     val = x[0]**2 + 3*x[0]*x[1] + 2*x[1]**2
    ...     score = np.array([2*x[0] + 3*x[1], 3*x[0] + 4*x[1]])
    ...     return val, score
    >>> x0 = np.array([1.0, 2.0])
    >>> G, Gt = gradient_2sided(test_func, x0)
    >>> print("Gradient G:", G)
    >>> print("Score matrix Gt:", Gt)
    """
    # Ensure x is a one-dimensional array.
    x = np.asarray(x).flatten()
    M = len(x)

    # Compute stepsize: h = eps^(1/3) * max(|x|, 1e-2)
    eps_val = np.finfo(float).eps ** (1 / 3)
    h = eps_val * np.maximum(np.abs(x), 1e-2)

    # Construct the "diagonal" perturbations (each column is e_i)
    ee = np.diag(h)

    # Test f at a perturbed x to see if it returns two outputs.
    test_out = f(x + ee[:, 0], *args, **kwargs)
    has_scores = isinstance(test_out, tuple) and (len(test_out) == 2)

    # Preallocate arrays for forward and backward evaluations.
    gf = np.zeros(M)
    gb = np.zeros(M)

    if has_scores:
        # Evaluate the first coordinate to get the length of the score vector.
        score_temp = np.atleast_1d(test_out[1])
        T = score_temp.size  # number of elements in the score vector
        Gf = np.zeros((T, M))
        Gb = np.zeros((T, M))

    # Compute the forward evaluations.
    for i in range(M):
        x_forward = x + ee[:, i]
        out = f(x_forward, *args, **kwargs)
        if has_scores:
            gf[i] = out[0]
            # Ensure that the score is a 1D array.
            Gf[:, i] = np.atleast_1d(out[1])
        else:
            gf[i] = out

    # Compute the backward evaluations.
    for i in range(M):
        x_backward = x - ee[:, i]
        out = f(x_backward, *args, **kwargs)
        if has_scores:
            gb[i] = out[0]
            Gb[:, i] = np.atleast_1d(out[1])
        else:
            gb[i] = out

    # Compute the central difference gradient.
    G = (gf - gb) / (2 * h)

    if has_scores:
        # Compute the matrix of individual scores:
        # For each coordinate j, the score is (forward score - backward score) divided by (2*h[j])
        denom = 2 * h  # shape (M,)
        Gt = (Gf - Gb) / denom[np.newaxis, :]  # broadcasting along rows
        return G, Gt
    else:
        return G


def hessian_2sided_nrows(func, x, K, *args, **kwargs):
    """
    Computes the last K rows of a two-sided finite-difference Hessian approximation.

    Parameters
    ----------
    func : callable
        A function f(x, *args, **kwargs) that accepts a 1D array x.
    x : array_like, shape (n,)
        Parameter vector.
    K : int
        Number of rows (from the end) of the Hessian to compute.
    *args, **kwargs :
        Additional arguments passed to func.

    Returns
    -------
    H_last : ndarray, shape (K, n)
        The last K rows of the approximate Hessian.

    This routine approximates the Hessian for the last K components using a two-sided finite difference:

       H(i,j) = [ f(x+e_i+e_j) - f(x+e_i) - f(x+e_j) + 2·f(x) - f(x-e_i) - f(x-e_j) + f(x-e_i-e_j) ]
                / [ 2 * h(i) * h(j) ]

    for i = n-K,..., n-1 and for j = 0,..., n-1.
    """
    x = np.asarray(x).flatten()
    n = len(x)
    if K > n:
        raise ValueError("K cannot be greater than the length of x")

    # Compute stepsize: h = eps^(1/3) * max(|x|, 1e-2)
    eps_val = np.finfo(float).eps ** (1 / 3)
    h = eps_val * np.maximum(np.abs(x), 1e-2)

    # Evaluate function at the base point
    fx = func(x, *args, **kwargs)

    # Compute forward (gp) and backward (gm) evaluations for each coordinate
    gp = np.zeros(n)
    gm = np.zeros(n)
    for i in range(n):
        x_forward = x.copy()
        x_forward[i] += h[i]
        gp[i] = func(x_forward, *args, **kwargs)

        x_backward = x.copy()
        x_backward[i] -= h[i]
        gm[i] = func(x_backward, *args, **kwargs)

    # Pre-allocate arrays for double forward (Hp) and double backward (Hm) evaluations.
    # We only need them for indices i in [n-K, n) (the last K components)
    Hp = np.empty((n, n))
    Hm = np.empty((n, n))
    # We'll fill only for i from n-K to n-1 and for all j=0,...,n-1.
    for i in range(n - K, n):
        for j in range(n):
            # Compute f(x + e_i + e_j)
            x_pp = x.copy()
            x_pp[i] += h[i]
            x_pp[j] += h[j]
            Hp[i, j] = func(x_pp, *args, **kwargs)

            # Compute f(x - e_i - e_j)
            x_mm = x.copy()
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]
            Hm[i, j] = func(x_mm, *args, **kwargs)
        # Symmetrize for safety (this loop covers only row i; in MATLAB code, symmetry is enforced)
        for j in range(n - K, n):
            Hp[j, i] = Hp[i, j]
            Hm[j, i] = Hm[i, j]

    # Precompute the outer product of stepsizes: hh(i,j)= h[i]*h[j]
    hh = np.outer(h, h)

    # Initialize full Hessian approximation (we only need the rows n-K ... n-1)
    H_full = np.zeros((n, n))
    for i in range(n - K, n):
        for j in range(n):
            # Use the formula from the MATLAB code:
            # H(i,j) = ( Hp(i,j) - gp(i) - gp(j) + 2*fx - gm(i) - gm(j) + Hm(i,j) ) / (2 * hh(i,j))
            H_full[i, j] = (Hp[i, j] - gp[i] - gp[j] + 2 * fx - gm[i] - gm[j] + Hm[i, j]) / (2 * hh[i, j])
            H_full[j, i] = H_full[i, j]  # enforce symmetry
    # Return only the last K rows
    return H_full[n - K: n, :]


# --- The main dcc function ---
# ---------------------------
# Updated DCC Estimation Function
# ---------------------------
def dcc(data, dataAsym, m, l, n, p=None, o=None, q=None, gjrType=None,
        method=None, composite=None, startingVals=None, options=None):
    """
    Updated Python implementation of the DCC estimation function using real
    univariate volatility estimation and DCC(1,1) log-likelihood following Engle (2002).

    INPUTS:
      data         - Either a (T x K) DataFrame or 2D array of zero-mean residuals.
      dataAsym     - Not used in this implementation.
      m, l, n      - For our DCC(1,1), we assume m=1, l=0, n=1.
      p, o, q, gjrType, startingVals, options - Univariate model options.

    OUTPUTS (as a dictionary):
      'parameters': final parameter vector (concatenated univariate parameters and DCC parameters),
      'll':         total log-likelihood,
      'Ht':         (K x K x T) array of conditional covariance matrices,
      'VCV':        (placeholder) covariance matrix of the parameters,
      'scores':     (placeholder) matrix of score vectors,
      'diagnostics': additional outputs (e.g., Q_bar).
    """
    # Convert data to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    T, k = data.shape

    data = 10 * data  # Scale the data by 10 (as per the original MATLAB code)

    # Set default orders for univariate volatility models if not provided.
    if p is None:
        p = np.ones(k, dtype=int)
    elif np.isscalar(p):
        p = p * np.ones(k, dtype=int)
    if o is None:
        o = np.zeros(k, dtype=int)
    elif np.isscalar(o):
        o = o * np.ones(k, dtype=int)
    if q is None:
        q = np.ones(k, dtype=int)
    elif np.isscalar(q):
        q = q * np.ones(k, dtype=int)
    if gjrType is None:
        gjrType = 2 * np.ones(k, dtype=int)
    elif np.isscalar(gjrType):
        gjrType = gjrType * np.ones(k, dtype=int)
    if options is None:
        options = {'disp': True, 'maxiter': 1000}

    # ---------------------------
    # Univariate Volatility Estimation
    # ---------------------------
    # data2d is the original T x k matrix of residuals.
    H_est, univariate = dcc_fit_variance(data, p, o, q, gjrType, startingVals)

    # ---------------------------
    # Estimate DCC parameters by maximizing the DCC log-likelihood.
    # ---------------------------
    # We assume a DCC(1,1) model here (so m=1 and n=1).
    res = minimize(lambda params: -dcc_likelihood(params, data, H_est),
                   x0=(0.01, 0.94),
                   bounds=((1e-6, 1), (1e-6, 1)),
                   options=options)
    params_dcc = res.x  # Estimated DCC parameters: [a, b]

    # ---------------------------
    # Reconstruct time-varying correlation matrices.
    # ---------------------------
    # Compute standardized residuals: z_t = data[t, :] / sqrt(H_est[t, :])
    z = np.zeros((T, k))
    for t in range(T):
        z[t, :] = data[t, :] / np.sqrt(H_est[t, :])
    Q_bar = np.cov(z, rowvar=False)
    Q = np.zeros((T, k, k))
    R_all = np.zeros((T, k, k))
    Q[0] = Q_bar.copy()
    for t in range(1, T):
        z_prev = z[t - 1, :].reshape(k, 1)
        Q[t] = (1 - params_dcc[0] - params_dcc[1]) * Q_bar + params_dcc[0] * (z_prev @ z_prev.T) + params_dcc[1] * Q[
            t - 1]
        diag_q = np.sqrt(np.diag(Q[t]))
        inv_diag = np.diag(1.0 / diag_q)
        R_all[t] = inv_diag @ Q[t] @ inv_diag
    # Set R_all[0] as the correlation matrix of Q_bar
    diag_q = np.sqrt(np.diag(Q_bar))
    inv_diag = np.diag(1.0 / diag_q)
    R_all[0] = inv_diag @ Q_bar @ inv_diag

    # ---------------------------
    # Reconstruct conditional covariance matrices: Ht[t] = diag(sqrt(H_est[t,:])) * R_all[t] * diag(sqrt(H_est[t,:]))
    Ht = np.zeros((k, k, T))
    for t in range(T):
        d_vec = np.sqrt(H_est[t, :])
        Ht[:, :, t] = np.outer(d_vec, d_vec) * R_all[t]

    # ---------------------------
    # Assemble Final Parameter Vector
    # ---------------------------
    # For the univariate part, simply take the conditional variance (at t=0) for each asset.
    garchParameters = H_est[0, :]
    parameters_final = np.concatenate((garchParameters, params_dcc))

    # Compute final log-likelihood from dcc_likelihood (using the estimated DCC parameters)
    ll_val = -dcc_likelihood(params_dcc, data, H_est)

    # Create dummy covariance of parameters and scores (for inference, these should be computed properly)
    v = len(parameters_final)
    VCV = np.eye(v) / T
    scores = np.zeros((T, v))
    diagnostics = {'Q_bar': Q_bar}

    return {'parameters': parameters_final,
            'll': ll_val,
            'Ht': Ht,
            'VCV': VCV,
            'scores': scores,
            'diagnostics': diagnostics}

def simulate_inefficiency(prices: pd.Series) -> pd.Series:
    # Le nombre de points correspond au nombre d'observations disponibles après changement et dropna.
    n = prices.pct_change().dropna().shape[0]
    dates = prices.pct_change().dropna().index
    # Simule des inefficiences uniformes entre -0.3 et 0.3.
    ineff_values = np.random.uniform(-0.3, 0.3, n)
    return pd.Series(ineff_values, index=dates)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Charger les données de prix et calculer les rendements (pour obtenir des séries "stationnaires").
    ftse100 = get_data('ftse100').pct_change().dropna()
    ftsemib = get_data('ftsemib').pct_change().dropna()
    sp500 = get_data('s&p500').pct_change().dropna()
    ssec = get_data('ssec').pct_change().dropna()

    # Pour cet exemple, nous simulons des inefficiences de façon à avoir des valeurs dans [-0.3, 0.3].
    ineff_ftse100 = simulate_inefficiency(ftse100).rename('FTSE100')
    ineff_ftsemib = simulate_inefficiency(ftsemib).rename('FTSEMIB')
    ineff_sp500   = simulate_inefficiency(sp500).rename('SP500')
    ineff_ssec    = simulate_inefficiency(ssec).rename('SSEC')

    # Combine the inefficiency series into a single DataFrame.
    ineff_df = pd.concat([ineff_ftse100, ineff_ftsemib, ineff_sp500, ineff_ssec], axis=1, join='inner')

    # For DCC estimation the residuals (inefficiency indices) should be zero-mean.
    ineff_df = ineff_df - ineff_df.mean()

    print("Inefficiency indices (first 5 rows):")
    print(ineff_df.head())

    # ------------------------------------------------
    # Estimate DCC(1,1) on the inefficiency indices.
    # ------------------------------------------------
    # In our blueprint dcc() function, we assume:
    #    data is a (T x K) DataFrame of zero-mean inefficiency indices.
    #    For a DCC(1,1) model we set m=1, l=0, and n=1.
    results = dcc(ineff_df, dataAsym=None, m=1, l=0, n=1)

    # Extract the estimated conditional covariance matrices Ht.
    Ht = results['Ht']
    k = ineff_df.shape[1]
    T = ineff_df.shape[0]

    # Compute conditional correlations from Ht by dividing Ht elementwise by outer(product of the standard deviations).
    Rt_est = np.zeros((k, k, T))
    for t in range(T):
        stds = np.sqrt(np.diag(Ht[:, :, t]))
        Rt_est[:, :, t] = Ht[:, :, t] / np.outer(stds, stds)

    # For example, plot the evolution of the estimated conditional correlation
    # between FTSE100 and FTSEMIB.
    cond_corr_ftse100_ftsemib = [Rt_est[0, 1, t] for t in range(T)]
    plt.figure(figsize=(10, 6))
    plt.plot(ineff_df.index, cond_corr_ftse100_ftsemib, label='FTSE100 & FTSEMIB')
    plt.xlabel("Date")
    plt.ylabel("Conditional Correlation")
    plt.title("Estimated DCC(1,1) Conditional Correlation between FTSE100 and FTSEMIB")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Estimated DCC Parameters:")
    print(results['parameters'])
    print("Log-likelihood:", results['ll'])
