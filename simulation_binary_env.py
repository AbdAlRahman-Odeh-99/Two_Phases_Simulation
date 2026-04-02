import numpy as np

from src.envs.binary_acq import BinaryAcquisitionEnvironment as Environment
from src.agents.binary_impute_agent import BinaryImputeAgent as Agent

print("Starting simulation...")

num_rounds_to_simulate = 5 # Number of rounds for this simulation run
total_agent_reward = 0
num_features = 5 # Define this explicitly for Environment and Agent
num_initial_observed_per_round = 1 # Define this here

env = Environment(num_features=num_features, num_total_instances=num_rounds_to_simulate)
agent = Agent(num_features=num_features)

env.reset(budget=10) # Initial reset with a defined budget

for i in range(num_rounds_to_simulate):
    print(f"\n--- Round {i+1} ---")
    # 1. The env gives an initial observation to the agent
    initial_observation, initial_observed_mask = env.get_observation() # Updated method name
    # 3. Based on imputed features, the agent decides to acquire some features
    # Pass environment's acquisition costs to the agent's decision function
    features_to_acquire_indices = agent.decide_acquisition(initial_observation=initial_observation,
                                                           initial_observed_mask=initial_observed_mask,
                                                           acquisition_costs=env.get_acquition_costs(),
                                                           budget_remains=env.get_remain_budget()) # Updated method name and parameters
    print(f"Agent decides to acquire features at indices: {features_to_acquire_indices}")

    newly_acquired_info, acquisition_cost, remaining_budget = env.acquire_feature(features_to_acquire_indices) # Updated method name
    print(f"Newly acquired features: {newly_acquired_info}, Cost: {acquisition_cost:.2f}, Remaining Budget: {remaining_budget:.2f}")
    prediction = agent.make_prediction(agent.imputed_features) # Use imputed features for prediction
    print(f"Agent makes prediction: {prediction}")
    #print(f"Environment reveals ground truth: {env.reveal_ground_truth()}") # Still useful for printing context
    round_reward, remaining_budget, terminated = env.step(prediction)
    print(f"Round Reward (prediction accuracy - cost): {round_reward:.2f}")
    total_agent_reward += round_reward

    # 6. The agent then updates its estimate based on the revealed information
    # Corrected parameter order for agent.update_estimates
    agent.update_estimates(newly_acquired_info, round_reward, prediction)
    print("Agent updates its internal models.")

print(f"\nSimulation finished. Total agent reward over {num_rounds_to_simulate} rounds: {total_agent_reward:.2f}")