"""
Example script demonstrating how to use the modularized repository.

This script shows:
1. Data generation
2. Phase 1 initialization
3. Phase 2 inference with different baselines
4. Result visualization and comparison
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    N_SAMPLES, K_CLUSTERS, M_MODALITIES, P_Y, RANDOM_SEED,
    T_PH1_LIST, HORIZON, COST_PER_MODALITY, BUDGET_FRACTION, FREE_VIEWS,
    KMEANS_RANDOM_STATE, KMEANS_N_INIT
)
from src.data import generate_synthetic_data
from src.models import initialization_phase
from src.utils import posterior_y, entropy, match_labels
from src.baselines import GreedyBaseline
from src.baselines.random import RandomBaseline
from src.visualization import plot_total_loss_vs_tph1


def example_basic_usage():
    """Example 1: Basic module usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Module Usage")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, Y, true_means, true_sigmas, rng = generate_synthetic_data(
        n_samples=N_SAMPLES,
        k_clusters=K_CLUSTERS,
        m_modalities=M_MODALITIES,
        p_y=np.array(P_Y),
        random_seed=RANDOM_SEED
    )
    print(f"   Data shape: {X.shape}, Labels: {Y.shape}")
    print(f"   Clusters: {K_CLUSTERS}, Modalities: {M_MODALITIES}")
    
    
    T_ph1 = T_PH1_LIST[0]
    # Initialize Phase 1
    print("\n2. Running Phase 1 initialization...")
    p_y_learned, learned_means, learned_sigmas, learned_centers = (
        initialization_phase(
            X[:T_ph1],
            K_CLUSTERS,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT
        )
    )
    print(f"   Phase 1 training samples: {T_ph1}")
    print(f"   Learned cluster priors: {p_y_learned}")
    print(f"   Learned means per modality: {list(learned_means.keys())}")
    
    # Make predictions on Phase 1
    print("\n3. Making predictions on Phase 1 data...")
    true_labels, y_hat = [], []
    for i in range(T_ph1):
        obs = {v: X[i, v] for v in range(M_MODALITIES)}
        p_post = posterior_y(obs, p_y_learned, learned_means, learned_sigmas)
        y_hat.append(np.argmax(p_post))
        true_labels.append(Y[i])
    
    y_hat_matched, _ = match_labels(true_labels, y_hat, K_CLUSTERS)
    correct = sum(1 if y_hat_matched[i] == true_labels[i] else 0 for i in range(len(true_labels)))
    accuracy_ph1 = correct / T_ph1
    print(f"   Phase 1 Accuracy: {accuracy_ph1:.3f}")
    

def example_baseline_comparison_one_sample():
    """Example 2: Compare different baselines."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Baseline Comparison")
    print("="*70)
    
    # Generate data
    X, Y, true_means, true_sigmas, rng = generate_synthetic_data(
        n_samples=5000,  # Smaller for faster example
        k_clusters=K_CLUSTERS,
        m_modalities=M_MODALITIES,
        p_y=np.array(P_Y),
        random_seed=RANDOM_SEED
    )
    
    # Initialize
    T_ph1 = 250
    p_y_learned, learned_means, learned_sigmas, learned_centers = (
        initialization_phase(X[:T_ph1], K_CLUSTERS)
    )
    
    # Test sample
    test_idx = T_ph1 + 10
    x_true = X[test_idx]
    budget = (HORIZON * (M_MODALITIES - len(FREE_VIEWS)) * BUDGET_FRACTION)
    initial_obs = {v: x_true[v] for v in FREE_VIEWS}
    
    # Greedy baseline
    print("\n1. Greedy Baseline (lambda_cost=0.1):")
    greedy = GreedyBaseline(
        p_y=p_y_learned,
        means=learned_means,
        sigmas=learned_sigmas,
        cost=COST_PER_MODALITY,
        free_views=FREE_VIEWS,
        lambda_cost=0.1
    )
    obs_greedy, budget_remaining_greedy, summary_greedy = greedy.acquire(
        x_true=x_true,
        budget=budget,
        initial_obs=initial_obs
    )
    print(f"   Acquired modalities: {sorted(obs_greedy.keys())}")
    print(f"   Budget spent: {budget - budget_remaining_greedy:.1f}/{budget:.1f}")
    print(f"   Acquisition decisions: {len(summary_greedy)}")
    
    # Random baseline
    print("\n2. Random Baseline:")
    random = RandomBaseline(
        p_y=p_y_learned,
        means=learned_means,
        sigmas=learned_sigmas,
        cost=COST_PER_MODALITY,
        free_views=FREE_VIEWS,
        random_seed=42
    )
    obs_random, budget_remaining_random, summary_random = random.acquire(
        x_true=x_true,
        budget=budget,
        initial_obs=initial_obs
    )
    print(f"   Acquired modalities: {sorted(obs_random.keys())}")
    print(f"   Budget spent: {budget - budget_remaining_random:.1f}/{budget:.1f}")
    print(f"   Acquisition decisions: {len(summary_random)}")
    
    # Compare predictions
    print("\n3. Prediction Comparison:")
    y_greedy, loss_greedy = greedy.predict(obs_greedy)
    y_random, loss_random = random.predict(obs_random)
    print(f"   Greedy predictions: y={y_greedy}, entropy={loss_greedy:.3f}")
    print(f"   Random prediction: y={y_random}, entropy={loss_random:.3f}")
    print(f"   True label: {Y[test_idx]}")

def example_configuration_modification():
    """Example 3: How to modify configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Configuration Modification")
    print("="*70)
    
    print("\nCurrent configuration in config.py:")
    print(f"  - Dataset size: {N_SAMPLES} samples")
    print(f"  - Clusters: {K_CLUSTERS}")
    print(f"  - Modalities: {M_MODALITIES}")
    print(f"  - Phase 1 sizes: {T_PH1_LIST}")
    print(f"  - Budget fraction: {BUDGET_FRACTION}")
    print(f"  - Free views: {FREE_VIEWS}")
    
    print("\nTo modify:")
    print("  1. Edit config.py directly")
    print("  2. Or import and override in your script:")
    print("     from config import N_SAMPLES")
    print("     N_SAMPLES = 10000  # Override")
    
    print("\nCommon modifications:")
    print("  - N_SAMPLES: Change dataset size")
    print("  - K_CLUSTERS: Change number of clusters")
    print("  - M_MODALITIES: Change number of modalities")
    print("  - BUDGET_FRACTION: Change budget per sample (fraction of M)")
    print("  - FREE_VIEWS: Change which modalities are free")
    print("  - LAMBDA_COST_LIST: Change cost parameters to test")
    print("  - T_PH1_LIST: Change Phase 1 training sizes")

def example_custom_baseline():
    """Example 4: Creating a custom baseline."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Baseline Template")
    print("="*70)
    
    print("""
To create a custom baseline:

1. Create src/baselines/my_baseline.py:
   
   from src.baselines import Baselines
   
   class MyBaseline(Baselines):
       def acquire(self, x_true, budget, initial_obs=None):
           # Your acquisition logic here
           observed = dict(initial_obs) if initial_obs else {}
           
           # ... your implementation ...
           
           return observed, budget_remaining, summary

2. Import and use:
   
   from src.baselines.my_baseline import MyBaseline
   
   baseline = MyBaseline(
       p_y=p_y_learned,
       means=learned_means,
       sigmas=learned_sigmas,
       cost=1.0,
       free_views=FREE_VIEWS
   )
   
   obs, budget_rem, summary = baseline.acquire(x_true, budget)

3. Current baselines available:
   - GreedyBaseline: Maximizes info gain minus cost
   - RandomBaseline: Random feature selection
    """)

def example_full_experiment_comparison():
    """Example 5: Running a full comparison experiment."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Full Experiment Comparison (Phase 2 Inference Accuracy)")
    print("="*70)
    
    # dataset for demo
    N_SAMPLES = 5000
    X, Y, true_means, true_sigmas, rng = generate_synthetic_data(
        n_samples=N_SAMPLES,
        k_clusters=K_CLUSTERS,
        m_modalities=M_MODALITIES,
        p_y=np.array(P_Y),
        random_seed=RANDOM_SEED
    )
    
    T_ph1 = 250
    budget_total = N_SAMPLES * BUDGET_FRACTION * (M_MODALITIES - len(FREE_VIEWS))
    budget_remaining = max(0,budget_total - (T_ph1 * COST_PER_MODALITY * (M_MODALITIES - len(FREE_VIEWS))))
    greedy_budget_remaining, random_budget_remaining = budget_remaining, budget_remaining

    # Initialize
    p_y_learned, learned_means, learned_sigmas, learned_centers = (
        initialization_phase(X[:T_ph1], K_CLUSTERS)
    )
    
    true_labels, predictions_greedy, predictions_random = [], [], []
    results_greedy = {"accuracies": [], "total_budget_spent": 0}
    results_random = {"accuracies": [], "total_budget_spent": 0}
    
    # Test Phase 2 samples
    for i in range(T_ph1, N_SAMPLES):
        x_true = X[i]
        initial_obs = {v: x_true[v] for v in FREE_VIEWS}
        
        # Greedy
        greedy = GreedyBaseline(
            p_y=p_y_learned, means=learned_means, sigmas=learned_sigmas,
            cost=COST_PER_MODALITY, free_views=FREE_VIEWS, lambda_cost=0.1
        )
        budget_remaining_before_acquisition = greedy_budget_remaining
        obs_g, greedy_budget_remaining, _ = greedy.acquire(x_true, greedy_budget_remaining, initial_obs)
        y_g, _ = greedy.predict(obs_g)
        y_true = Y[i]
        true_labels.append(y_true)
        predictions_greedy.append(y_g)
        results_greedy["total_budget_spent"] += (budget_remaining_before_acquisition - greedy_budget_remaining)
        
        # Random
        random = RandomBaseline(
            p_y_learned, learned_means, learned_sigmas,
            cost=COST_PER_MODALITY, free_views=FREE_VIEWS
        )

        budget_remaining_before_acquisition = random_budget_remaining
        obs_r, random_budget_remaining, _ = random.acquire(x_true, random_budget_remaining, initial_obs)
        y_r, _ = random.predict(obs_r)
        predictions_random.append(y_r)
        results_random["total_budget_spent"] += (budget_remaining_before_acquisition - random_budget_remaining)
        
    y_greedy_matched, _ = match_labels(true_labels, predictions_greedy, K_CLUSTERS)
    y_random_matched, _ = match_labels(true_labels, predictions_random, K_CLUSTERS)
    results_greedy["accuracies"] = list(1 if y_greedy_matched[i] == true_labels[i] else 0 for i in range(len(true_labels)))        
    results_random["accuracies"] = list(1 if y_random_matched[i] == true_labels[i] else 0 for i in range(len(true_labels)))        
    
    
    
    print(f"\nResults using {T_ph1} initialization samples and {N_SAMPLES - T_ph1} test samples:")
    print(f"  Greedy:  Accuracy={np.mean(results_greedy['accuracies']):.3f}, Budget Spent={results_greedy['total_budget_spent']}/{budget_remaining}")
    print(f"  Random:  Accuracy={np.mean(results_random['accuracies']):.3f}, Budget Spent={results_random['total_budget_spent']}/{budget_remaining}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_baseline_comparison_one_sample()
    example_configuration_modification()
    example_custom_baseline()
    example_full_experiment_comparison()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the README.md for detailed documentation")
    print("2. Run the full experiment: python experiments/two_phase_experiment.py")
    print("3. Create your own baseline in src/baselines/")
    print("4. Modify config.py for your experimental setup")
