"""
Acquisition policies for selective feature observation.
"""

import numpy as np
from typing import Dict, Set, Tuple, List
from src.utils.helpers import posterior_y, entropy, obs_vector


def distortion_loss(
    obs: Dict[int, float],
    centers: np.ndarray,
    p_post: np.ndarray
) -> float:
    """
    Compute expected distortion loss given observations.
    
    Args:
        obs: Observed modalities {modality_id: value}
        centers: Cluster centers (K, M)
        p_post: Posterior over clusters (K,)
        
    Returns:
        Expected distortion loss
    """
    x = obs_vector(obs, centers.shape[1])
    loss = 0.0
    for y, py in enumerate(p_post):
        loss += py * np.linalg.norm(x - centers[y]) ** 2
    return loss


def expected_information_gain(
    v: int,
    obs: Dict[int, float],
    p_y: np.ndarray,
    means: Dict[int, np.ndarray],
    sigmas: Dict[int, float]
) -> float:
    """
    Compute expected information gain from observing modality v.
    
    IG(Y; X_v | obs) using the learned joint distribution.
    
    Args:
        v: Modality index to evaluate
        obs: Current observations {modality_id: value}
        p_y: Cluster prior (K,)
        means: Per-modality means {m: (K,)}
        sigmas: Per-modality variances {m: float}
        
    Returns:
        Expected information gain
    """
    p_post = posterior_y(obs, p_y, means, sigmas)
    H_current = entropy(p_post)
    
    H_future = 0.0
    for y, py in enumerate(p_post):
        obs_new = dict(obs)
        obs_new[v] = means[v][y]  # conditional mean E[X^v|Y=y]
        p_post_future = posterior_y(obs_new, p_y, means, sigmas)
        H_future += py * entropy(p_post_future)
    
    return H_current - H_future


def greedy_acquisition_policy(
    x_true: np.ndarray,
    p_y: np.ndarray,
    means: Dict[int, np.ndarray],
    sigmas: Dict[int, float],
    lambda_cost: float,
    cost: float,
    budget: float,
    initial_obs: Dict[int, float] = None,
    free_views: Set[int] = None
) -> Tuple[Dict[int, float], float, List[Dict]]:
    """
    Greedy acquisition policy with budget constraints.
    
    Sequentially selects features to observe that maximize information gain
    minus the cost of observation, subject to budget constraints.
    
    Args:
        x_true: True feature values (M,)
        p_y: Cluster prior (K,)
        means: Per-modality means {m: (K,)}
        sigmas: Per-modality variances {m: float}
        lambda_cost: Cost weighting parameter
        cost: Cost per observation
        budget: Total budget remaining
        initial_obs: Initial observations (free observations)
        free_views: Set of modality indices that are always free
        
    Returns:
        observed: Final observations {modality_id: value}
        budget_remaining: Remaining budget
        summary: List of acquisition decisions
    """
    M = len(means)
    observed = dict(initial_obs) if initial_obs is not None else {}
    V = set(observed.keys())  # Set of acquired modalities
    
    if free_views is None:
        free_views = set()
    
    spent = 0.0
    summary = []
    
    while True:
        gains = {}
        
        # Evaluate all unobserved, non-free modalities
        for v in range(M):
            if v in V or v in free_views:
                continue
            
            ig = expected_information_gain(v, observed, p_y, means, sigmas)
            gains[v] = ig - lambda_cost * cost
        
        if not gains:
            break
        
        # Select modality with highest net gain
        v_star, best_gain = max(gains.items(), key=lambda x: x[1])
        
        # Check if acquisition is beneficial
        if best_gain <= 0 or spent + cost > budget:
            break
        
        # Acquire true value
        V.add(v_star)
        spent += cost
        observed[v_star] = x_true[v_star]
        budget = budget - spent
        
        summary.append({
            "chosen": v_star,
            "net_gain": best_gain,
            "V": set(V),
            "spent": spent
        })
    
    return observed, budget, summary
