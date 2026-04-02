import numpy as np
import itertools

def sigmoid(x):
    # numerically safe implementation
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
def predict_binary(parameters,features):
    return sigmoid(parameters.dot(features))

class BinaryImputeAgent:
    def __init__(self, num_features=10):
        self.num_features = num_features
        self.current_observation = None
        self.current_observed_mask = None
        self.imputed_features = None

        # naive method
        self.mask_to_order_map = {}
        self.masks_all =[]
        idx_views = list(range(1,self.num_features))
        for i in range(1,len(idx_views)+1):
            combos = itertools.combinations(idx_views, i)
            for combo in combos:
                self.masks_all.append(list(combo))
                self.mask_to_order_map[tuple(combo)] = len(self.masks_all)-1
        # Agent's internal models/parameters (placeholders)
        # Class-specific estimates (initially identical to global or more nuanced)
        self.class_feature_estimates = {
            0: {'counts': np.zeros(num_features), 'sum': np.zeros(num_features), 'sum_sq': np.zeros(num_features),
                'mean': np.zeros(num_features), 'std': np.ones(num_features) * 0.1},
            1: {'counts': np.zeros(num_features), 'sum': np.zeros(num_features), 'sum_sq': np.zeros(num_features),
                'mean': np.zeros(num_features), 'std': np.ones(num_features) * 0.1}
        }

        self.prediction_model_params = {'weights': np.random.rand(num_features), 'bias': 0.0}
        self.reward_cost_balancing_params = {'alpha': 2.0}

    def _observe(self, observation, observed_mask):
        self.current_observation = observation.copy()
        self.current_observed_mask = observed_mask.copy()
        print("[Debug] Agent observes: ", self.current_observation, " with mask: ", self.current_observed_mask)


    def _impute_features(self,class_to_impute=0):
        self.imputed_features = self.current_observation.copy()
        unobserved_indices = np.where(~self.current_observed_mask)[0] # Indices of features that are not observed
        
        for idx in unobserved_indices:
            # Confidence interval principle for imputation:
            class_mean = self.class_feature_estimates[class_to_impute]['mean'][idx]
            class_std = self.class_feature_estimates[class_to_impute]['std'][idx]
            class_count= self.class_feature_estimates[class_to_impute]['counts'][idx]
            # If there are very few observations for this feature, use a default imputation (e.g., prior mean)
            # Otherwise, sample from a normal distribution and clamp to a reasonable range.
            if class_count < 2: # Not enough data to reliably estimate std, use prior mean or a default
                self.imputed_features[idx] = class_mean # Or a default initial value
            else:
                # Sample from a normal distribution using estimated mean and std
                imputed_val = np.random.normal(class_mean, class_std)
                # Clamp the imputed value to a reasonable range (0-10) as features are generated in this range
                self.imputed_features[idx] = np.clip(imputed_val, -10, 10)
        return self.imputed_features

    def random_acquisition(self,initial_observation, initial_observed_mask, acquisition_costs, budget_remains):
        self._observe(initial_observation, initial_observed_mask) # Update agent's current observation and mask
        rnd_idx = np.random.randint(0,len(self.masks_all))
        selected_mask = self.masks_all[rnd_idx] # Randomly select one mask for acquisition
        # an internal imputation is needed to make the decision, but it is not used in this random acquisition strategy. In a more sophisticated strategy, the imputation would inform the acquisition decision.
        #self._observe(initial_observation, initial_observed_mask) # Update agent's current observation
        self._impute_features(class_to_impute=rnd_idx%2) # Impute features based on a randomly chosen class (0 or 1)
        return selected_mask
    
    def decide_acquisition(self,initial_observation, initial_observed_mask, acquisition_costs, budget_remains):
        self._observe(initial_observation, initial_observed_mask) # Update agent's current observation and mask
        
        unobserved_indices = np.where(~self.current_observed_mask)[0]
        if not unobserved_indices.size > 0:
            return []

        N_MONTE_CARLO_RUNS = 50 # Number of times to simulate and vote
        m_votes = []
        # start from each view that one do not have
        for i in range(N_MONTE_CARLO_RUNS):
            # impute class-wise evenly
            tc = i%2
            # full feature imputation
            self._impute_features(tc)
            # enumerate all combinations of view masks, except for the free indices
            # TODO:use submodularity property in some way to save complexity
            max_adjusted_reward = -np.inf
            max_adjusted_idx = None
            for m in self.masks_all:
                m_add = [0] + list(m) # adding the free observation, ensure m is a list for concatenation
                # m is the index set that the agent could request for
                tmp_pred = predict_binary(self.prediction_model_params['weights'][m_add], self.imputed_features[m_add]) # Fixed predict_binary call
                # compute expected reward...
                tmp_rew = tmp_pred**2 + (1-tmp_pred)**2 # FIXME: here is the reward structure
                tmp_cost = np.sum(acquisition_costs[m_add]) # Use np.sum for cost
                # record the expected reward
                adj_rew = tmp_rew - tmp_cost * self.reward_cost_balancing_params['alpha']
                if adj_rew > max_adjusted_reward:
                    max_adjusted_reward = adj_rew
                    max_adjusted_idx = m_add
            # record the decision of the best reward
            m_votes.append([max_adjusted_idx,max_adjusted_reward])
        # decide based on sorted max reward from class-wise samples
        # m_votes is a list of [mask, reward] pairs, we want to sort by reward and take the mask with the highest reward
        # create a dictionary to count the votes and the corresponding rewards for each feature across all Monte Carlo runs
        feature_vote_counts = {}
        feature_rewards = {}
        for item in m_votes:
            mask, reward = item
            ord_idx = self.mask_to_order_map[tuple(mask[1:])] # Get the order of the mask, default to 0 if not found
            if not ord_idx in feature_vote_counts:
                feature_vote_counts[ord_idx] = 0
                feature_rewards[ord_idx] = 0.0
            feature_vote_counts[ord_idx] += 1
            feature_rewards[ord_idx] += reward

        # Now we can compute the average reward for each feature and sort them
        sorted_features = sorted(feature_rewards.keys(), key=lambda x: feature_rewards[x]/feature_vote_counts[x], reverse=True)
        
        # implement a epsilon-greedy strategy to select features based on the average rewards
        epsilon = 0.1
        if np.random.rand() < epsilon:
            # Explore: randomly select a feature from the unobserved ones
            # select one from all possible masks uniformly at random
            rnd_idx = np.random.randint(0,len(self.masks_all))
            selected_mask = self.masks_all[rnd_idx] # Randomly select one mask for acquisition
            return selected_mask
        else:
            # Exploit: select the feature with the highest average reward that is still unobserved
            return self.masks_all[sorted_features[0]] # Return the mask with the highest average reward
        
    

    def make_prediction(self, features_for_prediction):
        # Placeholder: Simple linear model for binary classification
        # In a real scenario, this would be a trained classifier (e.g., Logistic Regression, SVM, Neural Net)
        # ignore those elements with np.nan, bias vector also need to adjust
        # take the bias vector from the non-nan elements of the features
        non_nan_indices = np.where(~np.isnan(features_for_prediction))[0]
        weighted_sum = np.dot(features_for_prediction[non_nan_indices], self.prediction_model_params['weights'][non_nan_indices]) \
                      + self.prediction_model_params['bias']

        probability = 1 / (1 + np.exp(-weighted_sum)) # Sigmoid
        prediction = 1 if probability > 0.5 else 0
        return prediction

    def update_estimates(self, revealed_features_info, reward, prediction):
        # Helper function to update statistics for a given set of estimates
        def _update_stats(estimates, idx, value):
            estimates['counts'][idx] += 1
            estimates['sum'][idx] += value
            estimates['sum_sq'][idx] += (value ** 2)

            count = estimates['counts'][idx]
            if count > 0:
                estimates['mean'][idx] = estimates['sum'][idx] / count
                if count > 1:
                    variance = (estimates['sum_sq'][idx] / count) - (estimates['mean'][idx] ** 2)
                    estimates['std'][idx] = np.sqrt(max(0, variance)) # Ensure std is not negative
                else:
                    estimates['std'][idx] = self.class_feature_estimates[0]['std'][idx] # Keep initial small std if only one observation (using class 0 as placeholder)

        # Update global feature distribution estimates
        # This part requires global_feature_estimates to be uncommented and initialized in Agent.__init__
        # For now, it's commented out in __init__ so this will not run unless uncommented.
        # If desired, add initialization for self.global_feature_estimates in Agent.__init__
        # for idx, value in revealed_features_info.items():
        #     _update_stats(self.global_feature_estimates, idx, value)

        true_y = prediction if reward == 1 else 1 - prediction # Infer true_y from prediction and reward
        # Update class-specific feature distribution estimates if true_y is revealed
        #if true_y is not None and true_y in self.class_feature_estimates:
        for idx, value in revealed_features_info.items():
            _update_stats(self.class_feature_estimates[true_y], idx, value)

        # Example: Dummy update for prediction model and balancing params
        # print(f"Agent updates: True Y={true_y}, Reward={reward}, Cost={acquisition_cost}")
        # In practice, this would involve training steps based on new data and feedback