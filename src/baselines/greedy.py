"""
Greedy acquisition baseline.
"""

import numpy as np
from typing import Dict, Set, Tuple, List
from src.baselines.base import Baselines
from src.acquisition.policies import greedy_acquisition_policy


class GreedyBaseline(Baselines):
    """
    Greedy acquisition baseline that maximizes information gain minus cost.
    """
    
    def __init__(
        self,
        p_y: np.ndarray,
        means: Dict[int, np.ndarray],
        sigmas: Dict[int, float],
        cost: float = 1.0,
        free_views: Set[int] = None,
        lambda_cost: float = 0.0
    ):
        """
        Initialize greedy baseline.
        
        Args:
            p_y: Cluster prior (K,)
            means: Per-modality means {m: (K,)}
            sigmas: Per-modality variances {m: float}
            cost: Cost per observation
            free_views: Set of modality indices that are always free
            lambda_cost: Cost weighting parameter
        """
        super().__init__(p_y, means, sigmas, cost, free_views)
        self.lambda_cost = lambda_cost
    
    def acquire(
        self,
        x_true: np.ndarray,
        budget: float,
        initial_obs: Dict[int, float] = None
    ) -> Tuple[Dict[int, float], float, List[Dict]]:
        """
        Acquire features using greedy policy.
        
        Args:
            x_true: True feature values (M,)
            budget: Total budget available
            initial_obs: Initial observations (free observations)
            
        Returns:
            observed: Final observations {modality_id: value}
            budget_remaining: Remaining budget
            summary: List of acquisition decisions
        """
        observed, budget_remaining, summary = greedy_acquisition_policy(
            x_true=x_true,
            p_y=self.p_y,
            means=self.means,
            sigmas=self.sigmas,
            lambda_cost=self.lambda_cost,
            cost=self.cost,
            budget=budget,
            initial_obs=initial_obs,
            free_views=self.free_views
        )
        
        return observed, budget_remaining, summary
