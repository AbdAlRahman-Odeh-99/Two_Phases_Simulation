"""
Configuration settings for the Two-Phase Simulation experiment.
"""

# Data generation parameters
N_SAMPLES = 5000
K_CLUSTERS = 2          # number of clusters
M_MODALITIES = 3        # number of modalities
D_MODALITY = 1          # dimensionality per modality (1D)

# Mixture weights
P_Y = [0.5, 0.5]

# Random seed
RANDOM_SEED = 0

# Experiment parameters
HORIZON = N_SAMPLES
COST_PER_MODALITY = 1
BUDGET_FRACTION = 0.4  # Total budget = Horizon * M * BUDGET_FRACTION

# Free views (always available without cost)
FREE_VIEWS = {0}

# Phase 1 configuration
T_PH1_LIST = list(range(250, HORIZON, 250))

# Phase 2 hyperparameters
LAMBDA_COST_LIST = [round(x, 2) for x in [i * 0.02 for i in range(11)]]  # 0.00 to 0.20

# KMeans parameters
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 10

# Numerical stability
EPS = 1e-12
