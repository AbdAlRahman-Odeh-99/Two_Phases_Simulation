# Two-Phases Simulation

A modularized repository for two-phase simulation experiments with support for multiple acquisition baselines and easy comparison.

## 📁 Project Structure

```
Two_Phases_Simulation/
├── config.py                           # Configuration parameters
├── README.md                           # This file
├── notebooks/
│   └── 2_Phases_Simulation_Free.ipynb  # Original notebook (reference)
├── src/
│   ├── __init__.py
│   ├── data/                          # Data generation
│   │   ├── __init__.py
│   │   └── data_generation.py
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── models/                        # Model initialization
│   │   ├── __init__.py
│   │   └── initialization.py
│   ├── acquisition/                   # Acquisition policies
│   │   ├── __init__.py
│   │   └── policies.py
│   ├── baselines/                     # Baseline implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── greedy.py
│   └── visualization/                 # Plotting utilities
│       ├── __init__.py
│       └── visualization.py
└── experiments/                       # Experiment scripts
    ├── __init__.py
    └── two_phase_experiment.py
```

## 🎯 Module Overview

### `config.py`
Central configuration file for all experiment parameters:
- Data generation settings (number of samples, clusters, modalities)
- Budget and cost parameters
- Random seeds
- Phase 1 length options and lambda cost values

### `src/data/`
**Data Generation Module**
- `generate_synthetic_data()`: Create synthetic data with true parameters
- `sample_joint_continuous()`: Sample from Gaussian mixture model

### `src/utils/`
**Helper Functions**
- `posterior_y()`: Compute posterior P(Y|X)
- `entropy()`: Entropy calculation
- `match_labels()`: Label matching using Hungarian algorithm
- `obs_vector()`: Convert observation dict to vector
- `conditional_entropy_y()`: Conditional entropy

### `src/models/`
**Model Learning**
- `initialization_phase()`: Learn model parameters from Phase 1 data using KMeans

### `src/acquisition/`
**Acquisition Policies**
- `expected_information_gain()`: Calculate information gain from observing a feature
- `greedy_acquisition_policy()`: Greedy policy with budget constraints
- `distortion_loss()`: Expected distortion loss calculation

### `src/baselines/`
**Baseline Implementations**
- `Baselines` (ABC): Abstract base class for all acquisition policies
- `GreedyBaseline`: Greedy information-gain maximization with cost tradeoff

### `src/visualization/`
**Plotting Functions**
- `plot_phase1_loss_vs_tph1()`: Phase 1 loss curves
- `plot_phase2_loss_vs_tph1()`: Phase 2 loss curves
- `plot_total_loss_vs_tph1()`: Combined loss curves
- `plot_accuracy_comparison()`: Accuracy comparison between phases

### `experiments/`
**Main Experiment Script**
- `two_phase_experiment.py`: Full pipeline for running experiments

## 🚀 Quick Start

### Running Experiments

```python
from experiments.two_phase_experiment import run_experiment

# Run experiments with default settings
results = run_experiment(
    baseline_name="greedy",
    lambda_costs=[0.0, 0.1, 0.2],
    verbose=True
)

# Access results
for lambda_cost, results_list in results.items():
    print(f"Lambda: {lambda_cost}, Results: {results_list}")
```

### Modifying Configuration

Edit `config.py` to change experiment parameters:

```python
# Example: Change Phase 1 sample sizes
T_PH1_LIST = list(range(100, 4000, 500))

# Change total budget
BUDGET_FRACTION = 0.5

# Change free views
FREE_VIEWS = {0, 1}  # First two modalities are free
```

## 🔧 Implementing New Baselines

To implement a new acquisition baseline:

### 1. Create a new baseline class

Create a file `src/baselines/my_baseline.py`:

```python
from src.baselines import Baselines
from typing import Dict, Set, Tuple, List
import numpy as np

class MyBaseline(Baselines):
    """Your custom baseline description."""
    
    def __init__(
        self,
        p_y: np.ndarray,
        means: Dict[int, np.ndarray],
        sigmas: Dict[int, float],
        cost: float = 1.0,
        free_views: Set[int] = None,
        **kwargs  # Any custom hyperparameters
    ):
        super().__init__(p_y, means, sigmas, cost, free_views)
        # Initialize custom parameters here
    
    def acquire(
        self,
        x_true: np.ndarray,
        budget: float,
        initial_obs: Dict[int, float] = None
    ) -> Tuple[Dict[int, float], float, List[Dict]]:
        """
        Implement your acquisition policy here.
        
        Returns:
            - observed: {modality_id: observed_value, ...}
            - budget_remaining: Unused budget
            - summary: List of acquisition decisions [{...}, ...]
        """
        observed = dict(initial_obs) if initial_obs is not None else {}
        budget_remaining = budget
        summary = []
        
        # Your acquisition logic here
        # Example: Randomly select features
        V = set(observed.keys())
        for v in range(self.M):
            if v not in V and v not in self.free_views:
                if budget_remaining >= self.cost:
                    observed[v] = x_true[v]
                    budget_remaining -= self.cost
                    V.add(v)
                    summary.append({"chosen": v})
        
        return observed, budget_remaining, summary
```

### 2. Update the experiment script

In `experiments/two_phase_experiment.py`, add your baseline:

```python
from src.baselines import MyBaseline

# In run_experiment()
if baseline_name.lower() == "my_baseline":
    baseline = MyBaseline(
        p_y=p_y_learned,
        means=learned_means,
        sigmas=learned_sigmas,
        cost=COST_PER_MODALITY,
        free_views=FREE_VIEWS,
        # Add custom parameters here
    )
```

### 3. Run experiments comparing baselines

```python
from experiments.two_phase_experiment import run_experiment

# Compare different baselines
greedy_results = run_experiment(baseline_name="greedy")
my_results = run_experiment(baseline_name="my_baseline")

# Analyze and compare results
```

## 📊 Experiment Components

### Phase 1: Initialization
- Uses Phase 1 data with all modalities observed
- Learns Gaussian model parameters via KMeans
- Estimates cluster priors and per-modality variances

### Phase 2: Inference
- Observes only free modalities initially
- Uses acquisition policy to select additional modalities
- Budget-constrained feature selection
- Computes predictions and losses

### Key Metrics
- `acc_ph1` / `acc_ph2`: Classification accuracy (after label matching)
- `ph1_prediction_loss` / `ph2_prediction_loss`: 1 - accuracy
- `post_loss_ph1` / `post_loss_ph2`: Entropy of posterior
- `dist_loss_ph1` / `dist_loss_ph2`: Expected distortion loss
- `total_prediction_loss`: Sum of Phase 1 and Phase 2 losses

## 🔌 Using Individual Modules

```python
# Data generation
from src.data import generate_synthetic_data
X, Y, true_means, true_sigmas, rng = generate_synthetic_data(
    n_samples=5000,
    k_clusters=2,
    m_modalities=3,
    p_y=np.array([0.5, 0.5]),
    random_seed=0
)

# Model initialization
from src.models import initialization_phase
p_y, learned_means, learned_sigmas, centers = initialization_phase(
    X[:2000], K=2
)

# Utility functions
from src.utils import posterior_y, entropy, match_labels
p_post = posterior_y(obs, p_y, learned_means, learned_sigmas)
h = entropy(p_post)

# Acquisition
from src.acquisition import expected_information_gain, distortion_loss
ig = expected_information_gain(v, obs, p_y, learned_means, learned_sigmas)

# Visualization
from src.visualization import plot_total_loss_vs_tph1
plot_total_loss_vs_tph1(results, lambda_cost=0.1)
```

## 📝 Changing Experiment Parameters

Key configuration options in `config.py`:

```python
# Dataset
N_SAMPLES = 5000              # Total dataset size
K_CLUSTERS = 2                # Number of clusters
M_MODALITIES = 3              # Number of modalities

# Budget
COST_PER_MODALITY = 1         # Cost per observation
BUDGET_FRACTION = 0.4         # Total budget = Horizon * M * BUDGET_FRACTION

# Phase 1 lengths
T_PH1_LIST = list(range(250, 5000, 250))

# Cost parameters
LAMBDA_COST_LIST = [0.0, 0.02, 0.04, ..., 0.20]

# Free modalities (always observed)
FREE_VIEWS = {0}
```

## 🛠️ Development

### Adding New Features
1. Add functions to appropriate module (`src/*/`)
2. Update module `__init__.py` with exports
3. Import and use in baselines or experiments

### Testing Baselines
```python
from src.baselines import GreedyBaseline
import numpy as np

# Create baseline
baseline = GreedyBaseline(
    p_y=np.array([0.5, 0.5]),
    means={0: np.array([[1.0], [-1.0]]), 1: np.array([[2.0], [-2.0]])},
    sigmas={0: 0.5, 1: 0.5},
    lambda_cost=0.1
)

# Test acquisition
obs, remaining, summary = baseline.acquire(
    x_true=np.array([1.5, 2.5, -0.5]),
    budget=5.0,
    initial_obs={0: 1.5}
)
```

## 📚 References

Original notebook: `notebooks/2_Phases_Simulation_Free.ipynb`

The modularized structure allows for:
- ✅ Easy baseline comparison
- ✅ Reproducible experiments
- ✅ Clean separation of concerns
- ✅ Extensible architecture
- ✅ Simple parameter tuning
