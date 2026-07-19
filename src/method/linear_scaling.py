from .core.bias_corrector import BiasCorrector
import numpy as np


class LinearScaling(BiasCorrector):
    """ additive == True for temperature (add); False for precipitation (multiply)"""
