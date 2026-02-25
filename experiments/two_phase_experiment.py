"""
Main two-phase simulation experiment.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    N_SAMPLES, K_CLUSTERS, M_MODALITIES, P_Y, RANDOM_SEED,
    HORIZON, COST_PER_MODALITY, BUDGET_FRACTION, FREE_VIEWS,
    T_PH1_LIST, LAMBDA_COST_LIST, KMEANS_RANDOM_STATE, KMEANS_N_INIT
)
from src.data import generate_synthetic_data
from src.models import initialization_phase
from src.utils import posterior_y, entropy, match_labels, obs_vector
from src.acquisition import distortion_loss
from src.baselines import GreedyBaseline
from src.visualization import (
    plot_phase1_loss_vs_tph1,
    plot_phase2_loss_vs_tph1,
    plot_total_loss_vs_tph1,
    plot_accuracy_comparison
)


def run_phase1_inference(
    X: np.ndarray,
    Y: np.ndarray,
    T_ph1: int,
    K: int,
    p_y: np.ndarray,
    learned_means: Dict,
    learned_sigmas: Dict,
    learned_centers: np.ndarray
) -> Dict:
    """
    Run inference on Phase 1 data (full modalities).
    
    Args:
        X: Full feature matrix
        Y: True labels
        T_ph1: Length of Phase 1
        K: Number of clusters
        p_y: Learned cluster priors
        learned_means: Learned means per modality
        learned_sigmas: Learned variances per modality
        learned_centers: Cluster centers
        
    Returns:
        Dictionary with Phase 1 metrics
    """
    M = X.shape[1]
    y_pred_ph1 = []
    post_loss_ph1 = []
    distortion_loss_ph1 = []
    
    for i in range(T_ph1):
        obs = {v: X[i, v] for v in range(M)}
        p_post = posterior_y(obs, p_y, learned_means, learned_sigmas)
        y_hat = np.argmax(p_post)
        y_pred_ph1.append(y_hat)
        post_loss_ph1.append(entropy(p_post))
        distortion_loss_ph1.append(distortion_loss(obs, learned_centers, p_post))
    
    y_pred_ph1 = np.array(y_pred_ph1)
    y_true_ph1 = Y[:T_ph1]
    y_pred_ph1_matched, _ = match_labels(y_true_ph1, y_pred_ph1, K)
    acc_ph1 = np.mean(y_pred_ph1_matched == y_true_ph1)
    
    return {
        "acc_ph1": acc_ph1,
        "ph1_prediction_loss": 1 - acc_ph1,
        "post_loss_ph1": np.mean(post_loss_ph1),
        "dist_loss_ph1": np.mean(distortion_loss_ph1)
    }


def run_phase2_inference(
    X: np.ndarray,
    Y: np.ndarray,
    T_ph1: int,
    Horizon: int,
    K: int,
    cost: float,
    Budget_total: float,
    Budget_remaining: float,
    p_y: np.ndarray,
    learned_means: Dict,
    learned_sigmas: Dict,
    learned_centers: np.ndarray,
    baseline,
    free_views: set
) -> Dict:
    """
    Run inference on Phase 2 data using acquisitions.
    
    Args:
        X: Full feature matrix
        Y: True labels
        T_ph1: Length of Phase 1
        Horizon: Total horizon
        K: Number of clusters
        cost: Cost per observation
        Budget_total: Total budget
        Budget_remaining: Remaining budget for Phase 2
        p_y: Learned cluster priors
        learned_means: Learned means per modality
        learned_sigmas: Learned variances per modality
        learned_centers: Cluster centers
        baseline: Baseline policy object
        free_views: Set of free modality indices
        
    Returns:
        Dictionary with Phase 2 metrics
    """
    T_ph2 = Horizon - T_ph1
    y_pred_ph2 = []
    post_loss_ph2 = []
    distortion_loss_ph2 = []
    
    for i in range(T_ph1, Horizon):
        initial_obs = {v: X[i, v] for v in free_views}
        obs, Budget_remaining, summary = baseline.acquire(
            x_true=X[i],
            budget=Budget_remaining,
            initial_obs=initial_obs
        )
        
        p_post = posterior_y(obs, p_y, learned_means, learned_sigmas)
        y_hat = np.argmax(p_post)
        y_pred_ph2.append(y_hat)
        post_loss_ph2.append(entropy(p_post))
        distortion_loss_ph2.append(distortion_loss(obs, learned_centers, p_post))
    
    y_pred_ph2 = np.array(y_pred_ph2)
    y_true_ph2 = Y[T_ph1:Horizon]
    y_pred_ph2_matched, _ = match_labels(y_true_ph2, y_pred_ph2, K)
    acc_ph2 = np.mean(y_pred_ph2_matched == y_true_ph2)
    
    return {
        "acc_ph2": acc_ph2,
        "ph2_prediction_loss": 1 - acc_ph2,
        "post_loss_ph2": np.mean(post_loss_ph2),
        "dist_loss_ph2": np.mean(distortion_loss_ph2)
    }


def run_experiment(
    baseline_name: str = "greedy",
    lambda_costs: List[float] = None,
    verbose: bool = True
) -> Dict[float, List[Dict]]:
    """
    Run complete two-phase experiment.
    
    Args:
        baseline_name: Name of baseline ("greedy", ...)
        lambda_costs: List of cost parameters to test
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping lambda_cost to list of results
    """
    if lambda_costs is None:
        lambda_costs = LAMBDA_COST_LIST
    
    # Generate synthetic data
    if verbose:
        print("Generating synthetic data...")
    X, Y, true_means, true_sigmas, rng = generate_synthetic_data(
        n_samples=N_SAMPLES,
        k_clusters=K_CLUSTERS,
        m_modalities=M_MODALITIES,
        p_y=np.array(P_Y),
        random_seed=RANDOM_SEED
    )
    
    # Convert P_Y to numpy array
    p_y = np.array(P_Y)
    
    # Budget calculation
    Budget_total = HORIZON * M_MODALITIES * BUDGET_FRACTION
    
    all_results = {}
    M = X.shape[1]
    
    for lambda_cost in lambda_costs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing with lambda_cost = {lambda_cost:.2f}")
            print(f"{'='*60}")
        
        results = []
        
        for T_ph1 in T_PH1_LIST:
            if verbose:
                print(f"  Running experiment with T_ph1 = {T_ph1}")
            
            # Phase 1: Initialization
            p_y_learned, learned_means, learned_sigmas, learned_centers = (
                initialization_phase(
                    X[:T_ph1],
                    K_CLUSTERS,
                    random_state=KMEANS_RANDOM_STATE,
                    n_init=KMEANS_N_INIT
                ) # NOTE: essentially do a kmeans here
            )
            
            # Phase 1 inference
            ph1_metrics = run_phase1_inference(
                X, Y, T_ph1, K_CLUSTERS,
                p_y_learned, learned_means, learned_sigmas, learned_centers
            )
            
            # Phase 2 budget
            Budget_remaining = Budget_total - (T_ph1 * COST_PER_MODALITY * (M_MODALITIES - len(FREE_VIEWS)))
            
            # Initialize baseline
            if baseline_name.lower() == "greedy":
                baseline = GreedyBaseline(
                    p_y=p_y_learned,
                    means=learned_means,
                    sigmas=learned_sigmas,
                    cost=COST_PER_MODALITY,
                    free_views=FREE_VIEWS,
                    lambda_cost=lambda_cost
                )
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")
            
            # Phase 2 inference
            ph2_metrics = run_phase2_inference(
                X, Y, T_ph1, HORIZON, K_CLUSTERS,
                COST_PER_MODALITY, Budget_total, Budget_remaining,
                p_y_learned, learned_means, learned_sigmas, learned_centers,
                baseline, FREE_VIEWS
            )
            
            # Aggregate metrics
            res = {
                "T_ph1": T_ph1,
                **ph1_metrics,
                **ph2_metrics,
                "total_prediction_loss": ph1_metrics["ph1_prediction_loss"] + ph2_metrics["ph2_prediction_loss"]
            }
            results.append(res)
            
            if verbose:
                print(
                    f"    Acc1={res['acc_ph1']:.3f}, "
                    f"Acc2={res['acc_ph2']:.3f}, "
                    f"TotalLoss={res['total_prediction_loss']:.3f}"
                )
        
        all_results[lambda_cost] = results
    
    return all_results


if __name__ == "__main__":
    # Run experiments
    results_dict = run_experiment(
        baseline_name="greedy",
        lambda_costs=LAMBDA_COST_LIST[:3],  # Test first 3 lambda values
        verbose=True
    )
    
    # Plot results
    print("\nGenerating plots...")
    for lambda_cost, results in results_dict.items():
        print(f"\nPlots for lambda_cost = {lambda_cost:.2f}")
        plot_phase1_loss_vs_tph1(results)
        plot_phase2_loss_vs_tph1(results)
        plot_total_loss_vs_tph1(results, lambda_cost)
        plot_accuracy_comparison(results)
