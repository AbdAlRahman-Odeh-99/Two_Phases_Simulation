"""
Base class for different acquisition baselines.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Set, Tuple, List


class Baselines(ABC):
    """Abstract base class for acquisition baselines."""
    
    def __init__(
        self,
        p_y: np.ndarray,
        means: Dict[int, np.ndarray],
        sigmas: Dict[int, float],
        cost: float = 1.0,
        free_views: Set[int] = None
    ):
        """
        Initialize baseline.
        
        Args:
            p_y: Cluster prior (K,)
            means: Per-modality means {m: (K,)}
            sigmas: Per-modality variances {m: float}
            cost: Cost per observation
            free_views: Set of modality indices that are always free
        """
        self.p_y = p_y
        self.means = means
        self.sigmas = sigmas
        self.cost = cost
        self.free_views = free_views if free_views is not None else set()
        self.M = len(means)
    
    @abstractmethod
    def acquire(
        self,
        x_true: np.ndarray,
        budget: float,
        initial_obs: Dict[int, float] = None
    ) -> Tuple[Dict[int, float], float, List[Dict]]:
        """
        Acquire features according to the baseline policy.
        
        Args:
            x_true: True feature values (M,)
            budget: Total budget available
            initial_obs: Initial observations (free observations)
            
        Returns:
            observed: Final observations {modality_id: value}
            budget_remaining: Remaining budget
            summary: List of acquisition decisions
        """
        pass
    
    def predict(self, obs: Dict[int, float]) -> Tuple[int, float]:
        """
        Make prediction given observations.
        
        Args:
            obs: Observed modalities {modality_id: value}
            
        Returns:
            y_hat: Predicted cluster
            loss: Prediction loss (entropy)
        """
        from src.utils.helpers import posterior_y, entropy
        
        p_post = posterior_y(obs, self.p_y, self.means, self.sigmas)
        y_hat = np.argmax(p_post)
        loss = entropy(p_post)
        return y_hat, loss
