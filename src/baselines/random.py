"""
Random acquisition baseline (example).
"""

import numpy as np
from typing import Dict, Set, Tuple, List
from src.baselines.base import Baselines


class RandomBaseline(Baselines):
    """
    Random acquisition baseline that randomly selects features within budget.
    
    Useful as a baseline for comparison.
    """
    
    def __init__(
        self,
        p_y: np.ndarray,
        means: Dict[int, np.ndarray],
        sigmas: Dict[int, float],
        cost: float = 1.0,
        free_views: Set[int] = None,
        random_seed: int = 0
    ):
        """
        Initialize random baseline.
        
        Args:
            p_y: Cluster prior (K,)
            means: Per-modality means {m: (K,)}
            sigmas: Per-modality variances {m: float}
            cost: Cost per observation
            free_views: Set of modality indices that are always free
            random_seed: Random seed for reproducibility
        """
        super().__init__(p_y, means, sigmas, cost, free_views)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
    
    def acquire(
        self,
        x_true: np.ndarray,
        budget: float,
        initial_obs: Dict[int, float] = None
    ) -> Tuple[Dict[int, float], float, List[Dict]]:
        """
        Acquire features using random selection policy.
        
        Args:
            x_true: True feature values (M,)
            budget: Total budget available
            initial_obs: Initial observations (free observations)
            
        Returns:
            observed: Final observations {modality_id: value}
            budget_remaining: Remaining budget
            summary: List of acquisition decisions
        """
        observed = dict(initial_obs) if initial_obs is not None else {}
        V = set(observed.keys())
        budget_remaining = budget
        summary = []
        
        # Get list of modalities that can be acquired
        acquirable = [v for v in range(self.M) if v not in V and v not in self.free_views]
        
        # Randomly shuffle
        self.rng.shuffle(acquirable)
        
        # Greedily acquire while budget allows
        for v in acquirable:
            if budget_remaining >= self.cost:
                observed[v] = x_true[v]
                budget_remaining -= self.cost
                V.add(v)
                summary.append({
                    "chosen": v,
                    "budget_spent": self.cost,
                    "budget_remaining": budget_remaining
                })
            else:
                break
        
        return observed, budget_remaining, summary
