"""
Data generation utilities for synthetic two-phase simulation.
"""

import numpy as np
from typing import Tuple, Dict


def sample_joint_continuous(
    N: int,
    p_y: np.ndarray,
    means: Dict[int, np.ndarray],
    sigmas: Dict[int, float],
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a Gaussian mixture model.
    
    Args:
        N: Number of samples
        p_y: Cluster prior probabilities (K,)
        means: Per-modality means {modality_id: (K, d)} 
        sigmas: Per-modality standard deviations {modality_id: float}
        rng: Random number generator
        
    Returns:
        X: Feature matrix (N, M) where M is number of modalities
        Y: Cluster labels (N,)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    K = len(p_y)
    M = len(means)
    
    # Sample cluster assignments
    Y = rng.choice(K, size=N, p=p_y)
    
    # Generate features for each sample and modality
    X = np.zeros((N, M))
    for i, y in enumerate(Y):
        for m in range(M):
            X[i, m] = rng.normal(
                loc=means[m][y, 0],
                scale=np.sqrt(sigmas[m])
            )
    
    return X, Y


def generate_synthetic_data(
    n_samples: int,
    k_clusters: int,
    m_modalities: int,
    p_y: np.ndarray,
    random_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, np.ndarray]:
    """
    Generate complete synthetic dataset with true parameters.
    
    Args:
        n_samples: Number of samples
        k_clusters: Number of clusters
        m_modalities: Number of modalities
        p_y: Cluster prior probabilities
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples, m_modalities)
        Y: True cluster labels (n_samples,)
        true_means: True cluster means per modality
        true_sigmas: True per-modality variances
        rng: Random number generator used
    """
    rng = np.random.default_rng(random_seed)
    d_m = 1  # 1D per modality
    
    # Generate true means for each modality
    true_means = {
        m: rng.normal(loc=0.0, scale=3.0, size=(k_clusters, d_m))
        for m in range(m_modalities)
    }
    
    # Generate per-modality variances
    true_sigmas = {
        m: 0.5 + rng.random()
        for m in range(m_modalities)
    }
    
    # Sample data
    X, Y = sample_joint_continuous(
        N=n_samples,
        p_y=p_y,
        means=true_means,
        sigmas=true_sigmas,
        rng=rng
    )
    
    return X, Y, true_means, true_sigmas, rng
