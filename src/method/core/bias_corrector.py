from abc import ABC, abstractmethod

import numpy as np


class BiasCorrector(ABC):
    """
    Manipulating 1D numpy arrays over time at the point data stage.
    Designing xarray.apply_ufunc compatibility to extend to grid data

    at Stage 7 without rewriting the core algorithm.
    """

    @abstractmethod
    def fit(self, obs: np.ndarray, sim_hist: np.ndarray) -> "BiasCorrector":
        """Learn the parameter calibration in historical (obs compare with sim_hist)"""

    @abstractmethod
    def correct(self, sim: np.ndarray) -> np.ndarray:
        """Apply correction for any simulation"""
