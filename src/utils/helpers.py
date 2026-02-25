"""
Helper functions for the two-phase simulation.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def obs_vector(obs: Dict[int, float], M: int) -> np.ndarray:
    """
    Convert observation dictionary to vector.
    
    Args:
        obs: Dictionary {modality_id: value}
        M: Total number of modalities
        
    Returns:
        Vector representation (M,)
    """
    x = np.zeros(M)
    for m, v in obs.items():
        x[m] = v
    return x


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute entropy: $H(X) = -\\sum p(x) \\log p(x)$
    
    Args:
        p: Probability distribution
        eps: Small constant for numerical stability
        
    Returns:
        Entropy value
    """
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def posterior_y(
    obs: Dict[int, float],
    p_y: np.ndarray,
    means: Dict[int, np.ndarray],
    sigmas: Dict[int, float]
) -> np.ndarray:
    """
    Compute posterior distribution: $p(Y | x^{(\\mathcal{V})})$
    
    Args:
        obs: Observed modalities {modality_id: value}
        p_y: Prior on clusters (K,)
        means: Per-modality cluster means {m: (K,)}
        sigmas: Per-modality variances {m: float}
        
    Returns:
        Posterior probabilities (K,)
    """
    logp = np.log(p_y + 1e-12)
    
    for v, x_v in obs.items():
        mu = means[v]  # shape: (K,)
        var = sigmas[v]
        logp += -0.5 * ((x_v - mu) ** 2) / var
        logp += -0.5 * np.log(2 * np.pi * var)
    
    # Numerical stability
    logp -= np.max(logp)
    p = np.exp(logp)
    return p / p.sum()


def conditional_entropy_y(
    obs: Dict[int, float],
    p_y: np.ndarray,
    means: Dict[int, np.ndarray],
    sigmas: Dict[int, float]
) -> float:
    """
    Compute conditional entropy: $H(Y | x^{(\\mathcal{V})})$
    
    Args:
        obs: Observed modalities
        p_y: Prior on clusters
        means: Per-modality cluster means
        sigmas: Per-modality variances
        
    Returns:
        Conditional entropy value
    """
    p_post = posterior_y(obs, p_y, means, sigmas)
    return entropy(p_post)


def match_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Match predicted labels to true labels using Hungarian algorithm.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        K: Number of clusters
        
    Returns:
        y_pred_matched: Matched predictions
        label_map: Mapping from predicted to true labels
    """
    # Confusion matrix: rows = true labels, cols = predicted labels
    C = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    
    # Hungarian algorithm (maximize total agreement)
    row_ind, col_ind = linear_sum_assignment(-C)
    
    # Build mapping: predicted -> true
    label_map = {pred: true for true, pred in zip(row_ind, col_ind)}
    
    # Apply mapping
    y_pred_matched = np.array([label_map[y] for y in y_pred])
    
    return y_pred_matched, label_map
