import numpy as np
import itertools

class BinaryAcquisitionEnvironment:
    def __init__(self, num_features=10, num_total_instances=100, acquisition_costs=None, budget=10):
        self.num_features = num_features
        self.num_total_instances = num_total_instances
        self.budget = budget

        # Define Gaussian Mixture Model parameters for feature generation
        self.num_components = 2 # For example, two components
        self.component_means = np.array([
            np.random.rand(num_features) * 5,  # Component 0: lower values
            np.random.rand(num_features) * 5 + 5 # Component 1: higher values
        ])
        self.component_stds = np.array([
            np.random.rand(num_features) * 0.5 + 0.5, # Component 0: some std
            np.random.rand(num_features) * 0.5 + 0.5  # Component 1: some std
        ])

        # Generate all true features and ground truth labels globally
        self.all_true_features = []
        self.all_ground_truth_y = []
        for _ in range(self.num_total_instances):
            chosen_component_idx = np.random.randint(0, self.num_components)
            instance_true_features = np.random.normal(self.component_means[chosen_component_idx],
                                                      self.component_stds[chosen_component_idx])
            instance_ground_truth_y = np.random.choice([0, 1])
            self.all_true_features.append(instance_true_features)
            self.all_ground_truth_y.append(instance_ground_truth_y)

        self.all_true_features = np.array(self.all_true_features)
        self.all_ground_truth_y = np.array(self.all_ground_truth_y)

        # Define acquisition costs for each feature
        self.acquisition_costs = acquisition_costs if acquisition_costs is not None else np.random.rand(num_features) * 0.5
        self.acquisition_costs[0] = 0 # initial observation is for free, but cannot be acquired again

        # Initialize instance-specific variables, which will be truly set in reset()
        self.true_features = None
        self.ground_truth_y = None
        self.observed_mask = None
        self.features_current_state = None # this is the masked feature
        self.budget_remains = budget # This will be reset in reset()
        self.current_instance_idx = 0

    def get_observation(self):
        return self.features_current_state.copy(), self.observed_mask.copy()

    def get_remain_budget(self):
        return self.budget_remains

    def get_acquition_costs(self):
        return self.acquisition_costs.copy()

    def acquire_feature(self, feature_indices_to_acquire):
        if self.current_instance_idx >= self.num_total_instances:
            print("[Warning] acquire_feature called after all instances have been processed. No acquisition possible.")
            return {}, 0.0, self.budget_remains
        acquisition_cost = 0.0
        newly_acquired_features = {}
        # check system budget limit
        # Note: The original code had a bug here, it should be sum of costs for the *selected* features, not all.
        # This check is more accurate for the actual cost of acquisition for the current indices.
        cost_of_selected_features = np.sum(self.acquisition_costs[list(feature_indices_to_acquire)])
        if self.budget_remains < cost_of_selected_features:
          # If budget is not enough for ANY of the selected features, acquire none.
          # Or, one could implement a greedy approach to acquire as many as possible within budget.
          # For simplicity, returning empty if budget is insufficient for the requested set.
          return newly_acquired_features, acquisition_cost, self.budget_remains

        for idx in feature_indices_to_acquire:
            if not self.observed_mask[idx]:
                self.observed_mask[idx] = True
                self.features_current_state[idx] = self.true_features[idx]
                acquisition_cost += self.acquisition_costs[idx]
                newly_acquired_features[idx] = self.true_features[idx]

        # private budget deduction
        self.budget_remains -= acquisition_cost

        return newly_acquired_features, acquisition_cost, self.budget_remains

    def _get_reward_and_cost(self, agent_prediction):
        if self.current_instance_idx >= self.num_total_instances:
            # warning that the system has terminated, 
            print("[Warning] Environment step called after all instances have been processed. Returning 0 reward.")
            return 0.0
        # NOTE: the instance index is handled by outer simulation loop,
        # NOTE: the loop should avoid repeating prediction and hance cheating on accumulating reward
        # Reward based on prediction accuracy and penalized by acquisition cost
        prediction_accuracy_reward = 1 if agent_prediction == self.ground_truth_y else 0
        total_reward = prediction_accuracy_reward
        return total_reward

    # Never reveal the ground truth directly.
    # Let the agent decide
    def reveal_ground_truth(self):
        return self.ground_truth_y
        
    def reset(self,budget=100):
        # reset the environment
        # instance back to 0
        self.current_instance_idx = 0
        # budget refill
        self.budget_remains = budget
        # reset the observed mask and current state
        self.observed_mask = np.zeros(self.num_features, dtype=bool)
        self.observed_mask[0] = True # The first feature is always observed for free
        #self.features_current_state = np.zeros(self.num_features)
        self.features_current_state = np.full(self.num_features, np.nan) # Use NaN to indicate unobserved features more clearly
        # fill in the free initial observation
        self.features_current_state[0] = self.all_true_features[self.current_instance_idx][0]
        # set the true features and ground truth for the current instance
        self.true_features = self.all_true_features[self.current_instance_idx]
        self.ground_truth_y = self.all_ground_truth_y[self.current_instance_idx]
        # update the current state with the initially observed feature
        self.features_current_state[0] = self.true_features[0]

    def step(self, prediction):
        # print current index for debugging
        print(f"[Debug] Environment step: Current instance index: {self.current_instance_idx}")
        #newly_acquired_info, acquisition_cost, remaining_budget = self.acquire_feature(features_to_acquire_indices)
        reward = self._get_reward_and_cost(prediction)
        print(f"Environment step: Prediction={prediction}, Ground Truth={self.ground_truth_y}, Reward={reward:.2f}")
        remaining_budget = self.get_remain_budget()
        # we need to update instance index now as the loop only manage time horizon
        self.current_instance_idx += 1
        if self.current_instance_idx >= self.num_total_instances:
            return reward, remaining_budget, True # episode done
        self.true_features = self.all_true_features[self.current_instance_idx]
        self.ground_truth_y = self.all_ground_truth_y[self.current_instance_idx]
        self.observed_mask = np.zeros(self.num_features, dtype=bool)
        self.observed_mask[0] = True # The first feature is always observed for free
        #self.features_current_state = np.zeros(self.num_features)
        self.features_current_state = np.full(self.num_features, np.nan) # Use NaN to indicate unobserved features more clearly
        self.features_current_state[0] = self.all_true_features[self.current_instance_idx][0]

        return reward, remaining_budget, False