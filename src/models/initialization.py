"""
Model initialization and learning.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple


def initialization_phase(
    X: np.ndarray,
    K: int,
    random_state: int = 0,
    n_init: int = 10
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, float], np.ndarray]:
    """
    Initialize model parameters using KMeans clustering on Phase 1 data.
    
    Args:
        X: Training data (T_ph1, M)
        K: Number of clusters
        random_state: Random seed for KMeans
        n_init: Number of KMeans initializations
        
    Returns:
        p_y: Learned cluster priors (K,)
        learned_means: Learned means per modality {m: (K,)}
        learned_sigmas: Learned variances per modality {m: float}
        learned_centers: Cluster centers (K, M)
    """
    T_ph1, M = X.shape
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    learned_centers = kmeans.cluster_centers_  # (K, M)
    
    # Cluster prior
    p_y = np.bincount(labels, minlength=K) / T_ph1
    
    # Per-modality parameters
    learned_means = {}
    learned_sigmas = {}
    
    for v in range(M):
        learned_means[v] = learned_centers[:, v]
        learned_sigmas[v] = np.var(X[:, v]) + 1e-6
    
    return p_y, learned_means, learned_sigmas, learned_centers
