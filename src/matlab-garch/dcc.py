from abstract_mgarch import AbstractMGARCH
import matlab.engine
import numpy as np


class DCC(AbstractMGARCH):

    def __init__(self):
        super().__init__()

    def run(self, data: np.ndarray, p: int = 1, q: int = 1, type_flag: int = 1, nargout: int = 6):
        
        parameters, ll, Ht, VCV, scores, diagnostics = self.engine.dcc(matlab.double(data),
                                                                       [],
                                                                       matlab.double([p]),
                                                                       matlab.double([q]),
                                                                       matlab.double([type_flag]),
                                                                       nargout)
        
        var_1_dcc = np.squeeze(Ht[0, 0, :])
        var_2_dcc = np.squeeze(Ht[1, 1, :])
        cov_1_2_dcc = np.squeeze(Ht[1, 0, :])
        corr_1_2_dcc = cov_1_2_dcc / np.sqrt(var_1_dcc * var_2_dcc)
        
        return corr_1_2_dcc

