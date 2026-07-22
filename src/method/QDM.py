from .core.bias_corrector import BiasCorrector
import numpy as np


class QuantileDeltaMapping(BiasCorrector):
    """
    Quantile Delta Mapping — Cannon, Sobie, & Murdock (2015).
    ---
    Quantile TAU compute the quantile in Sim_future (Not compute in Sim_hist),
    so this method can keep the delta change in GCMs projection. Even if future 
    """