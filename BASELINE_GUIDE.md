"""
Guide: Implementing New Acquisition Baselines

This document provides step-by-step instructions for creating new acquisition
baselines to compare against existing methods.
"""

# =============================================================================
# QUICK START: 5-Minute Implementation
# =============================================================================

"""
Step 1: Create src/baselines/my_baseline.py with a class inheriting from Baselines

from src.baselines import Baselines
import numpy as np
from typing import Dict, Set, Tuple, List

class MyAcquisitionPolicy(Baselines):
    
    def acquire(self, x_true, budget, initial_obs=None):
        observed = dict(initial_obs) if initial_obs else {}
        budget_remaining = budget
        summary = []
        
        # Your acquisition logic here
        # Pick features that maximize your objective within budget
        
        return observed, budget_remaining, summary


Step 2: Add to experiments/two_phase_experiment.py

if baseline_name.lower() == "my_baseline":
    baseline = MyAcquisitionPolicy(...)


Step 3: Run experiments

results = run_experiment(baseline_name="my_baseline")
"""

# =============================================================================
# DETAILED TEMPLATE WITH COMMON PATTERNS
# =============================================================================

"""
from src.baselines import Baselines
from src.acquisition import expected_information_gain
import numpy as np
from typing import Dict, Set, Tuple, List

class CustomBaseline(Baselines):
    \"\"\"
    Custom acquisition baseline.
    
    Description:
    - What does this baseline do?
    - How does it select features?
    - What parameters does it use?
    \"\"\"
    
    def __init__(
        self,
        p_y: np.ndarray,
        means: Dict[int, np.ndarray],
        sigmas: Dict[int, float],
        cost: float = 1.0,
        free_views: Set[int] = None,
        # Add custom hyperparameters
        my_param: float = 0.5
    ):
        \"\"\"Initialize baseline with parameters.\"\"\"
        super().__init__(p_y, means, sigmas, cost, free_views)
        self.my_param = my_param
    
    def acquire(
        self,
        x_true: np.ndarray,
        budget: float,
        initial_obs: Dict[int, float] = None
    ) -> Tuple[Dict[int, float], float, List[Dict]]:
        \"\"\"
        Acquire features according to your policy.
        
        Args:
            x_true: True feature values (M,) for one sample
            budget: Budget available for acquisitions
            initial_obs: Already-observed modalities (free observations)
        
        Returns:
            observed: Dictionary of {modality_id: observed_value}
            budget_remaining: Unused budget
            summary: List of acquisition decisions with metadata
        \"\"\"
        # Initialize observations with free views
        observed = dict(initial_obs) if initial_obs is not None else {}
        V = set(observed.keys())
        
        budget_remaining = budget
        summary = []
        
        # Main acquisition loop
        while True:
            # Find best feature to acquire
            best_feature = None
            best_score = -float('inf')
            
            for v in range(self.M):
                # Skip already observed or free modalities
                if v in V or v in self.free_views:
                    continue
                
                # Your scoring function here
                score = your_scoring_function(v, observed)
                
                if score > best_score:
                    best_score = score
                    best_feature = v
            
            # Check stopping conditions
            if best_feature is None:
                break
            if best_score <= 0:  # Not worth acquiring
                break
            if budget_remaining < self.cost:  # Out of budget
                break
            
            # Acquire the feature
            V.add(best_feature)
            observed[best_feature] = x_true[best_feature]
            budget_remaining -= self.cost
            
            summary.append({
                "chosen": best_feature,
                "score": best_score,
                "budget_spent": self.cost,
                "budget_remaining": budget_remaining
            })
        
        return observed, budget_remaining, summary
    
    def your_scoring_function(self, v, obs):
        \"\"\"Implement your acquisition strategy here.\"\"\"
        # Example options:
        # 1. Information gain: expected_information_gain(v, obs, self.p_y, self.means, self.sigmas)
        # 2. Uncertainty reduction: entropy(posterior_y(...))
        # 3. Cost-adjusted: ig - self.my_param * self.cost
        pass
"""

# =============================================================================
# COMMON BASELINE PATTERNS
# =============================================================================

"""
PATTERN 1: Information Gain Maximization
- Acquire features that most reduce uncertainty about the label
- Use: expected_information_gain(v, obs, p_y, means, sigmas)

from src.acquisition import expected_information_gain

score = expected_information_gain(v, observed, self.p_y, self.means, self.sigmas)


PATTERN 2: Uncertainty Sampling
- Acquire features to reduce posterior entropy
- Use: posterior_y() and entropy()

from src.utils import posterior_y, entropy

p_post = posterior_y(observed, self.p_y, self.means, self.sigmas)
uncertainty = entropy(p_post)


PATTERN 3: Feature Importance
- Pre-compute or learn feature importance
- Acquire high-importance features first

# Example: variance-based importance
importance = {}
for v in range(self.M):
    importance[v] = np.var(self.means[v])


PATTERN 4: Cost-Benefit Analysis
- Use objective: benefit - lambda * cost
- Vary lambda to trade off informativeness vs cost

score = information_gain - self.lambda_cost * self.cost


PATTERN 5: Diversity-Based
- Acquire features that provide diverse information
- Combine multiple information sources


PATTERN 6: Budget-Aware Greedy
- Adaptive strategy based on remaining budget
- Adjust greediness as budget depletes

remaining_ratio = budget_remaining / total_budget
threshold = baseline_threshold * remaining_ratio
"""

# =============================================================================
# USING AVAILABLE UTILITIES
# =============================================================================

"""
from src.utils.helpers import:
  - posterior_y(obs, p_y, means, sigmas) -> posterior probabilities
  - entropy(p) -> entropy of distribution
  - obs_vector(obs, M) -> convert dict to vector
  
from src.acquisition.policies import:
  - expected_information_gain(v, obs, p_y, means, sigmas)
  - distortion_loss(obs, centers, p_post)
  - greedy_acquisition_policy(...) -> reference implementation
  
from src.utils.helpers import:
  - match_labels(y_true, y_pred, K) -> align predictions to labels
  - conditional_entropy_y(obs, p_y, means, sigmas)
"""

# =============================================================================
# COMPARISON WITH GREEDY BASELINE
# =============================================================================

"""
The default GreedyBaseline uses:

    score = IG(v) - lambda_cost * cost
    
    Where:
    - IG(v) = H(Y|obs) - H(Y|obs ∪ {x_v})  [information gain]
    - lambda_cost: cost parameter (varies in experiments)
    - cost: acquisition cost per feature

To implement a variant:
1. Modify the scoring function
2. Add hyperparameters to control behavior
3. Test with different configurations
"""

# =============================================================================
# TESTING YOUR BASELINE
# =============================================================================

"""
1. Unit test with a single sample:

from src.baselines.my_baseline import MyBaseline
from src.data import generate_synthetic_data
from src.models import initialization_phase
import numpy as np

# Generate and initialize
X, Y, _, _, _ = generate_synthetic_data(1000, 2, 3, np.array([0.5, 0.5]))
p_y, means, sigmas, _ = initialization_phase(X[:200], K=2)

# Test baseline
baseline = MyBaseline(p_y, means, sigmas)
x_test = X[250]
obs, budget_rem, summary = baseline.acquire(x_test, budget=5.0)
print(f"Acquired: {obs}, Budget remaining: {budget_rem}")

2. Compare with greedy:

greedy = GreedyBaseline(p_y, means, sigmas, lambda_cost=0.1)
obs_greedy, _, _ = greedy.acquire(x_test, budget=5.0)

print(f"Your baseline: {sorted(obs.keys())}")
print(f"Greedy: {sorted(obs_greedy.keys())}")

3. Run full experiment comparison:

results_yours = run_experiment(baseline_name="my_baseline")
results_greedy = run_experiment(baseline_name="greedy")

# Compare metrics in results_yours vs results_greedy
"""

# =============================================================================
# ADDING CUSTOM HYPERPARAMETERS
# =============================================================================

"""
1. Add to your __init__:

class MyBaseline(Baselines):
    def __init__(self, ..., my_hyperparam=0.5, another_param=2):
        super().__init__(...)
        self.my_hyperparam = my_hyperparam
        self.another_param = another_param

2. Modify config.py or pass when instantiating:

baseline = MyBaseline(
    p_y=p_y_learned,
    means=learned_means,
    sigmas=learned_sigmas,
    cost=COST_PER_MODALITY,
    free_views=FREE_VIEWS,
    my_hyperparam=0.8,
    another_param=3
)

3. Test different values:

for my_hp in [0.3, 0.5, 0.7]:
    baseline = MyBaseline(..., my_hyperparam=my_hp)
    results = run_experiment(...)  # Use your baseline
"""

# =============================================================================
# PUBLISHING YOUR BASELINE
# =============================================================================

"""
Once your baseline is working:

1. Add to src/baselines/ with descriptive name
2. Update src/baselines/__init__.py
3. Add comparison in experiments/two_phase_experiment.py
4. Document in README.md with:
   - Strategy description
   - When to use it
   - Expected performance characteristics
5. Provide usage example
"""

# =============================================================================
# DEBUGGING TIPS
# =============================================================================

"""
1. Add print statements to understand acquisitions:
   - Which features are selected?
   - What are the scores?
   - How is budget being spent?

2. Check that summary is properly formatted:
   - Each element should be a dict
   - Include relevant metadata

3. Verify budget accounting:
   - budget_remaining + spent = initial budget
   - Test edge cases (empty obs, full budget, no budget)

4. Compare predictions:
   - Use baseline.predict(observed) to get predictions
   - Check that entropy is lower with more observations

5. Profile performance if needed:
   - Use cProfile for timing
   - Optimize hot loops
"""

if __name__ == "__main__":
    print("This is a guide file. See examples.py for runnable code.")
