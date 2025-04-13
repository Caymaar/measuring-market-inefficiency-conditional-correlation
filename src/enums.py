from enum import Enum
from .hurst_estimation import *

class HurstMethodType(Enum):
    """
    Define the mapping between an estimation method and its implementation.
    """
    SCALED_VARIANCE = ScaledVariance
    SCALED_WINDOWED_VARIANCE = ScaledWindowedVariance
    AGGREGATE_VARIANCE = ...
    DFA_ANALYSIS = ...
    RS_ANALYSIS = ...
