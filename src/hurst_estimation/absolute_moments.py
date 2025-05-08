import numpy as np
from scipy.stats import linregress
from .abstract_estimation import AbstractHurstEstimator


class AbsoluteMomentsEstimator(AbstractHurstEstimator):
    """
    Estimateur de Hurst par la méthode des moments absolus généralisés
    """
    
    def __init__(self, time_series: np.ndarray, q: float = 1.0, tau_min: int = 1, tau_max: int = None, num_tau: int = 20):
        """
        Parameters:
            q: Ordre du moment (1 par défaut pour les moments absolus)
            tau_min: Délai minimal 
            tau_max: Délai maximal (demi-longueur de la série par défaut)
            num_tau: Nombre de délais à analyser
        """
        super().__init__(time_series)
        self.q = q
        self.tau_min = tau_min
        self.tau_max = tau_max if tau_max else len(time_series) // 2
        self.num_tau = num_tau

    def estimate(self) -> float:
        ts = self.ts
        n = len(ts)
        q = self.q
        
        taus = np.logspace(np.log10(self.tau_min), 
                          np.log10(self.tau_max),
                          num=self.num_tau,
                          base=10).astype(int)
        taus = np.unique(taus[taus < n//2])
        
        moments = []
        log_taus = []
        
        for tau in taus:
            if tau >= n:
                continue
                
            diffs = np.abs(ts[tau:] - ts[:-tau])
            
            if len(diffs) == 0:
                continue
                
            m = np.nanmean(diffs ** q)
            
            if m <= 0:
                continue
                
            moments.append(m)
            log_taus.append(np.log(tau))
            
        if len(log_taus) < 2:
            raise ValueError("Pas assez de points pour la régression")
            
        slope, _, _, _, _ = linregress(log_taus, np.log(moments))
        
        return slope / q