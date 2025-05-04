from enum import Enum
from .hurst_estimation import *
from .garch import *


class HurstMethodType(Enum):
    """
    Define the mapping between an estimation method and its implementation.
    """
    SCALED_VARIANCE = ScaledVariance
    SCALED_WINDOWED_VARIANCE = ScaledWindowedVariance
    AGGREGATE_VARIANCE = AggregateVarianceHurstEstimator
    ABSOLUTE_MOMENTS = AbsoluteMomentsHurstEstimator
    DFA_ANALYSIS = DfaHurstEstimator
    HIGUCHI_ANALYSIS = HiguchiHurstEstimator
    TTA_ANALYSIS = TtaHurstEstimator
    RS_ANALYSIS = ...


class GarchMethodType(Enum):
    """
    Define the mapping between a multivariate garch method and its implementation.
    """
    DCC = mgarch
    BEKK = ...
    VECH = ...
    FULL_VECH = ...
    DIAG_VECH = ...
