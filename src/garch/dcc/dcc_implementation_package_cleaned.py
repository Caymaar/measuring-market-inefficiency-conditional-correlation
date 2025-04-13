import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv, sqrtm
from scipy.special import gamma
import sys
sys.path.append('.')  # Ajout du répertoire parent au chemin d'importation
from src.utils import get_data


# ---------------------------
# HurstCalculator (tel que fourni)
# ---------------------------
class HurstCalculator:
    def __init__(self, k=10, window=250):
        self.k = k
        self.window = window

    def get_hurst_exponent(self, time_series):
        lags = range(2, self.k)
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return reg[0]

    def rolling_hurst(self, prices: pd.Series) -> pd.Series:
        log_prices = np.log(prices)
        hurst_values = []
        index_list = []
        for i in range(len(log_prices) - self.window + 1):
            window_data = log_prices.iloc[i:i + self.window].values
            h = self.get_hurst_exponent(window_data)
            hurst_values.append(h)
            index_list.append(log_prices.index[i + self.window - 1])
        return pd.Series(hurst_values, index=index_list)

    def calculate_inefficiency(self, prices: pd.Series) -> pd.Series:
        hurst_series = self.rolling_hurst(prices)
        inefficiency_series = 0.5 - hurst_series
        return inefficiency_series


# ---------------------------
# Classe mgarch mise à jour avec vraisemblance DCC réelle
# ---------------------------
class mgarch:
    def __init__(self, dist='norm'):
        if dist in ['norm', 't']:
            self.dist = dist
        else:
            raise ValueError("Le paramètre 'dist' doit être 'norm' ou 't'")

    def garch_fit(self, returns):
        # Estime un modèle GARCH(1,1) (ici une implémentation simple)
        res = minimize(self.garch_loglike, (0.01, 0.01, 0.94), args=(returns,),
                       bounds=((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        # Log-vraisemblance pour les innovations normales
        LogL = np.sum(-np.log(2 * np.pi * var_t)) - np.sum((returns.A1 ** 2) / (2 * var_t))
        return -LogL

    def garch_var(self, params, returns):
        T = len(returns)
        omega, alpha, beta = params
        var_t = np.zeros(T)
        for i in range(T):
            if i == 0:
                var_t[i] = returns[i] ** 2
            else:
                var_t[i] = omega + alpha * (returns[i - 1] ** 2) + beta * var_t[i - 1]
        return var_t

    def dcc_loglike_real(self, params, D_t):
        """
        Calcule la log-vraisemblance du modèle DCC(1,1) pour des innovations normales.

        params : tuple (a, b) pour la dynamique DCC
        D_t    : Matrice (T x N) des écarts-types conditionnels estimés des modèles univariés.

        On part de z_t = r_t/D_t (résidus standardisés) et on utilise :
            Q_t = (1 - a - b) * Q_bar + a * (z_{t-1} z_{t-1}ᵀ) + b * Q_{t-1}
            R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        La contribution à la vraisemblance de l'instant t est :
            l_t = -0.5 [ ln(det(R_t)) + z_tᵀ R_t^{-1} z_t ]
        La vraisemblance totale est la somme sur t.
        """
        a, b = params
        z = np.array(self.rt) / D_t  # Division élément par élément ; z est de taille (T,N)
        T, N = z.shape
        Q_bar = np.cov(z, rowvar=False)
        Q_t = np.zeros((T, N, N))
        R_t = np.zeros((T, N, N))
        ll = 0.0
        Q_t[0] = Q_bar.copy()
        # Boucle sur t=1,...,T-1 (en prenant z_t de t=1,...)
        for t in range(1, T):
            z_prev = z[t - 1, :].reshape(N, 1)
            Q_t[t] = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q_t[t - 1]
            diag_Q = np.sqrt(np.diag(Q_t[t]))
            # Inverse des racines carrées de la diagonale
            inv_diag = np.diag(1.0 / diag_Q)
            R_t[t] = inv_diag @ Q_t[t] @ inv_diag
            # Calcul du log-déterminant de R_t
            sign, logdet = np.linalg.slogdet(R_t[t])
            if sign <= 0:
                return 1e6  # Pénalité en cas de problème numérique
            term = np.dot(z[t, :], np.linalg.solve(R_t[t], z[t, :]))
            ll += -0.5 * (logdet + term)
        return ll  # On maximise la vraisemblance (ou on minimise son opposé)

    def fit(self, returns):
        """
        Estime le modèle mgarch sur les retours (dans notre cas, les indices d’inefficience) :
        - L’ensemble des retours (self.rt) est décentré.
        - Pour chaque série, on estime un modèle GARCH(1,1) afin d’obtenir D_t.
        - Ensuite, on estime les paramètres DCC a et b en maximisant la log-vraisemblance du modèle DCC.
        """
        # Ici, returns doit être une structure (ex. array) 2D de dimension (T x N)
        self.rt = np.matrix(returns)
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        if self.N == 1 or self.T == 1:
            raise ValueError("Il faut un array 2D avec plus de 1 colonne")
        # Décentrer les séries
        self.mean = self.rt.mean(axis=0)
        self.rt = self.rt - self.mean

        # Estimation des écarts-types conditionnels par modèle univarié (GARCH(1,1))
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            params = self.garch_fit(self.rt[:, i])
            D_t[:, i] = np.sqrt(self.garch_var(params, self.rt[:, i]))
        self.D_t = D_t

        # Estimation des paramètres du DCC (stade 2)
        # On maximise la vraisemblance en minimisant son opposé.
        res = minimize(lambda params: -self.dcc_loglike_real(params, D_t),
                       x0=(0.01, 0.94),
                       bounds=((1e-6, 1), (1e-6, 1)))
        self.a, self.b = res.x

        return {'mu': self.mean, 'alpha': self.a, 'beta': self.b}

    def predict(self, ndays=1):
        """
        Prédit la matrice de covariance conditionnelle pour ndays.
        (Pour simplifier, cette fonction retourne simplement la covariance conditionnelle
        prédites pour le dernier instant multipliée par sqrt(ndays).)
        """
        if not hasattr(self, 'a'):
            print("Modèle non estimé.")
            return None
        Q_bar = np.cov(np.array(self.rt) / self.D_t, rowvar=False)
        Q_t = np.zeros((self.T, self.N, self.N))
        R_t = np.zeros((self.T, self.N, self.N))
        H_t = np.zeros((self.T, self.N, self.N))
        Q_t[0] = Q_bar.copy()
        for i in range(1, self.T):
            dts = np.diag(self.D_t[i])
            dtinv = inv(dts)
            et = dtinv * self.rt[i].T
            Q_t[i] = (1 - self.a - self.b) * Q_bar + self.a * (et @ et.T) + self.b * Q_t[i - 1]
            diag_q = np.sqrt(np.diag(Q_t[i]))
            inv_diag = np.diag(1.0 / diag_q)
            R_t[i] = inv_diag @ Q_t[i] @ inv_diag
            H_t[i] = dts @ R_t[i] @ dts
        if self.dist == 'norm':
            return {'dist': self.dist, 'cov': H_t[-1] * np.sqrt(ndays)}
        elif self.dist == 't':
            return {'dist': self.dist, 'dof': self.dof, 'cov': H_t[-1] * np.sqrt(ndays)}
        else:
            return None


# ---------------------------
# Main : utilisation des données réelles et simulation d'une estimation DCC sur l'inefficience
# ---------------------------
if __name__ == "__main__":
    # def get_data(market_name):
    #     # Remplacer cette fonction par votre chargement de données réel.
    #     # Ici, nous simulons des prix via un mouvement brownien géométrique.
    #     np.random.seed(42)
    #     T = 500
    #     dates = pd.date_range(start='2020-01-01', periods=T, freq='D')
    #     S0 = 100
    #     mu = 0.0002
    #     sigma = 0.01
    #     returns = np.random.normal(mu, sigma, T)
    #     prices = S0 * np.exp(np.cumsum(returns))
    #     return pd.Series(prices, index=dates)
    # Charger les données et transformer en rendements
    ftse100 = get_data('ftse100')
    ftsemib = get_data('ftsemib')
    sp500 = get_data('s&p500')
    ssec = get_data('ssec')

    def simulate_inefficiency(series):
        n = series.shape[0]
        dates = series.index
        ineff_values = np.random.uniform(-0.3, 0.3, n)
        return pd.Series(ineff_values, index=dates)


    ineff_ftse100 = simulate_inefficiency(ftse100).rename('FTSE100')
    ineff_ftsemib = simulate_inefficiency(ftsemib).rename('FTSEMIB')
    ineff_sp500 = simulate_inefficiency(sp500).rename('SP500')
    ineff_ssec = simulate_inefficiency(ssec).rename('SSEC')

    # Concaténer les indices d'inefficience dans un DataFrame (sur les dates communes)
    ineff_df = pd.concat([ineff_ftse100, ineff_ftsemib, ineff_sp500, ineff_ssec], axis=1, join='inner')
    # Décentrer les séries (moyenne nulle)
    ineff_df = ineff_df - ineff_df.mean()
    print("Inefficiency indices (first 5 rows):")
    print(ineff_df.head())

    # Utiliser le modèle mgarch pour estimer la dynamique conditionnelle sur ces inefficiences.
    ineff_matrix = ineff_df.values  # forme (T x N)

    model = mgarch(dist='norm')
    fit_results = model.fit(ineff_matrix)
    print("Estimated mgarch (DCC) parameters:")
    print(fit_results)

    # Pour visualiser la dynamique, on peut récupérer la matrice de covariance conditionnelle (via predict)
    prediction = model.predict(ndays=1)
    print("Predicted conditional covariance matrix:")
    print(prediction)

    # Vous pouvez également reconstruire les matrices de corrélation conditionnelle à partir de H_t :
    # (ici, nous supposons que la fonction predict retourne la dernière covariance conditionnelle)

    # Par exemple, tracer la corrélation conditionnelle entre FTSE100 et FTSEMIB
    T_effective = ineff_matrix.shape[0]
    # Pour tracer l'évolution, on refait une boucle en utilisant la dynamique du modèle DCC.
    z = np.array(model.rt) / model.D_t  # résidus standardisés (T x N)
    Q_bar = np.cov(z, rowvar=False)
    Q_t = np.zeros((T_effective, model.N, model.N))
    R_t = np.zeros((T_effective, model.N, model.N))
    Q_t[0] = Q_bar.copy()
    for t in range(1, T_effective):
        z_prev = z[t - 1, :].reshape(model.N, 1)
        Q_t[t] = (1 - model.a - model.b) * Q_bar + model.a * (z_prev @ z_prev.T) + model.b * Q_t[t - 1]
        diag_Q = np.sqrt(np.diag(Q_t[t]))
        inv_diag = np.diag(1.0 / diag_Q)
        R_t[t] = inv_diag @ Q_t[t] @ inv_diag
    # Extraire, par exemple, la corrélation entre FTSE100 (colonne 0) et FTSEMIB (colonne 1)
    cond_corr = [R_t[t][0, 1] for t in range(T_effective)]

    plt.figure(figsize=(10, 6))
    plt.plot(ineff_df.index, cond_corr, label='FTSE100 & FTSEMIB')
    plt.xlabel("Date")
    plt.ylabel("Conditional Correlation")
    plt.title("DCC(1,1) : Conditional Correlation between FTSE100 and FTSEMIB")
    plt.legend()
    plt.tight_layout()
    plt.show()
