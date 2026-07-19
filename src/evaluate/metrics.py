import numpy as np


def mean_bias(obs: np.ndarray, sim: np.ndarray) -> float:
    """ Mean bias (ME) - suitable for evaluating the temperature"""
    return float(np.mean(sim - obs))

def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """ Percent bias (%) - suitable for evaluating the precipitation, sensitive to total volume"""
    return float(100 * np.sum(sim - obs) / np.sum(obs))

def wet_day_frequency(obs: np.ndarray, sim: np.ndarray, threshold: float = 1.0) -> float:
    """ Proportion of rainy days (>= Threshold mm) - detects 'drizzle problem' """
    return float(np.mean(sim > threshold))
