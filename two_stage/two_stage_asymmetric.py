import csv
import time
from itertools import combinations
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import linear_sum_assignment

from core.lp_colgen import solve_lp_policy_colgen, sample_lp_colgen_policy

def match_cluster_labels(learned_centers, true_centers):
    # Convert dict of arrays (modality -> (K,1)) to (K,M)
    true_mean_array = np.hstack([true_centers[m] for m in sorted(true_centers.keys())])
    K = learned_centers.shape[0]
    
    # Cost matrix: squared Euclidean distances
    cost_matrix = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i,j] = np.sum((learned_centers[i] - true_mean_array[j])**2)
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return mapping


def generate_data_blobs(rng, nsamples=1000, nclasses=2, nviews=10, seed=42):
    mm = rng.random(size=(nclasses, nviews)) * 2  # heterogeneous view means
    stds = 1.0
    data = make_blobs(nsamples, n_features=nviews, centers=mm,
                       cluster_std=stds, random_state=seed)
    return data[0], mm, data[1]


def generate_synthetic_data(rng, n_samples, k_clusters, m_modalities, p_y, random_seed=0):
    X, mm, Y = generate_data_blobs(rng, n_samples, k_clusters, m_modalities, random_seed)
    true_means = {m: mm[:, m].reshape(-1, 1) for m in range(m_modalities)}
    true_sigmas = {m: 1.0 for m in range(m_modalities)}
    return X, Y, true_means, true_sigmas, rng


def generate_per_view_costs(m_modalities, rng):
    costs = rng.random(size=(m_modalities,))
    costs[0] = 0
    costs = costs / np.sum(costs)
    return costs


def generate_combination_costs_heterogeneous(view_combinations, per_view_costs):
    return {combo: sum(per_view_costs[v - 1] for v in combo) for combo in view_combinations}


def generate_view_combinations(m_modalities):
    modalities = list(range(2, m_modalities + 1))
    all_views = []
    for r in range(len(modalities) + 1):
        for combo in combinations(modalities, r):
            all_views.append((1,) + combo)
    return all_views

def predict_combo(x_sample, centers, combo, return_score=False):
    """K=2 nearest-center prediction restricted to the observed (combo)
    views. return_score: if True, also return a continuous "evidence for
    class 1" score (distances[0] - distances[1]; >0 means center 1 is
    closer, i.e. matches the hard prediction pred==1). AUROC needs this
    continuous score, not just the hard argmin -- same convention used by
    gmm_2class_bandit_asymmetric.py / gmm_2class_submodular_asymmetric.py's
    predict_with_observed(return_score=True)."""
    mask = np.zeros(len(x_sample), dtype=bool)
    mask[np.array(combo) - 1] = True
    distances = np.sum((centers[:, mask] - x_sample[mask]) ** 2, axis=1)
    pred = int(np.argmin(distances))
    if return_score:
        score = float(distances[0] - distances[1])
        return pred, score
    return pred

# DUPLICATE FUNCTION (for now) - we can unify later
def predict_single_combination(x_sample, centers, combo, return_score=False):
    mask = np.zeros(len(x_sample), dtype=bool)
    mask[np.array(combo) - 1] = True
    distances = np.sum((centers[:, mask] - x_sample[mask]) ** 2, axis=1)
    pred = np.argmin(distances)
    if return_score:
        score = float(distances[0] - distances[1])
        return pred, score
    return pred

def initialize_centers(x, y, n_init_samples, k_clusters, m_modalities, rng):
    """
    Online initialization using labeled samples.

    Centers start as k_clusters distinct rows of the ACTUAL DATA x, chosen
    via rng.choice (without replacement) -- IDENTICAL convention to
    gmm_2class_bandit_asymmetric.py's est_means_init /
    gmm_2class_submodular_asymmetric.py's est_means_init. Unlike an earlier
    version of this function, that rng.choice draw is now a genuine
    BLENDED PRIOR (like the other two scripts' est_counts=1 dummy-count
    convention), not something immediately overwritten on the first sample
    of a class -- see the FIX note below.

    For every incoming sample in the loop over the first n_init_samples:
        1. Predict using the CURRENT centers (nearest-center, all
           k_clusters centers always valid from round 0 onward).
        2. Record the prediction error.
        3. Update the true class's center via a supervised running mean
           (the revealed true label yi selects WHICH center updates --
           same convention as gmm_2class_bandit_asymmetric.py's
           run_phase1_training: est_counts starts at 1 representing the
           rng.choice seed as a pseudo-observation, blended in via a
           `1 - 1/(count+1)` scaler rather than a hard overwrite).

    FIX (previously an inconsistency vs. the other two scripts): this used
    to hard-overwrite learned_centers[yi] = xi the FIRST time each class
    was seen, discarding the rng.choice seed for that class almost
    immediately (typically within the first 1-2 samples, since p(class)
    is usually not tiny) -- making the initial rng.choice draw functionally
    dead code, unlike bandit/submodular where est_means_init genuinely
    seeds the running mean from round 1. Now it doesn't: the rng.choice
    seed is always blended in with weight 1/(count+1), exactly like the
    other two scripts, instead of being thrown away.

    Note this does NOT introduce any label-matching ambiguity (unlike an
    initial concern one might have): the update is supervised by the
    REVEALED TRUE LABEL yi at every step (learned_centers[yi] is what gets
    updated, never a predicted/nearest label), so center index k is
    guaranteed-by-construction to track true class k throughout, exactly
    as before this fix and exactly like gmm_2class_bandit_asymmetric.py /
    gmm_2class_submodular_asymmetric.py's est_means[y_true, ...] update.
    That's also why this function's caller uses an IDENTITY label_map
    rather than Hungarian-matching -- see two_stage_runner.py's
    run_inference_lp_dataset docstring.

    Returns
    -------
    learned_centers : ndarray, shape (k_clusters, m_modalities)
        Learned class centers.

    init_error : float
        Online prediction error over all n_init_samples rounds (every
        round now makes a real prediction, unlike before where the first
        occurrence of each class skipped prediction entirely).
    """

    # n_seed_rows = min(k_clusters, len(x))
    # seed_rows = rng.choice(len(x), size=n_seed_rows, replace=False)
    # learned_centers = x[seed_rows[np.arange(k_clusters) % n_seed_rows]].astype(float).copy()
    learned_centers = rng.normal(loc=0.0, scale=1.0, size=(2, m_modalities))

    # Dummy prior count of 1 (matches gmm_2class_bandit_asymmetric.py's
    # est_counts = np.ones(...) convention) -- the rng.choice seed counts
    # as pseudo-observation #1 for its cluster, blended down as real
    # observations arrive, rather than being discarded outright.
    counts = np.ones(k_clusters, dtype=int)

    mistakes = 0
    predictions = 0

    for i in range(n_init_samples):
        xi = x[i]
        yi = y[i]

        # Predict using the CURRENT centers -- all k_clusters centers are
        # always valid from round 0 onward now (seeded by rng.choice), so
        # there's no more an "active_centers" subset to restrict to.
        dists = np.linalg.norm(learned_centers - xi, axis=1)
        pred = np.argmin(dists)

        predictions += 1
        if pred != yi:
            mistakes += 1

        # Supervised running-mean update using the REVEALED TRUE LABEL yi
        # -- same formula as gmm_2class_bandit_asymmetric.py's
        # run_phase1_training (scaler = 1 - 1/(count+1), applied BEFORE
        # incrementing count).
        scaler = 1 - 1 / (counts[yi] + 1)
        learned_centers[yi] = scaler * learned_centers[yi] + (1 - scaler) * xi
        counts[yi] += 1

    init_error = mistakes / predictions if predictions > 0 else 0.0

    return learned_centers, init_error

# This version of EXP4 is adapted to handle the budget constraint using a primal-dual approach inspired by BwK algorithms.
# Additionally, it includes a safe fallback policy when the budget is exhausted, which is a common technique in constrained bandit settings.
# Finally, it updates the weights of all experts in a full-information manner at each step.
def run_alg(x, y, centers, view_combinations, combo_costs, T1, training_budget, rng):
    total_samples = len(x)
    T2 = total_samples - T1
    if T2 <= 0:
        return {}

    n_experts = len(view_combinations)

    # EXP4 params (aligned with references: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf)
    gamma = min(1.0, np.sqrt((n_experts * np.log(n_experts)) / max(1, T2)))
    eta = gamma / n_experts

    # Dual learning rate
    alpha = 1.0 / np.sqrt(max(1, T2))

    # Initialize
    weights = np.ones(n_experts) ### Reward estimates from the stage 1 learned centers
    lambda_t = 0.0

    training_budget_spent = float(T1)
    initial_remaining_budget = float(training_budget - T1)
    training_remaining_budget = initial_remaining_budget

    predictions = []
    scores = []
    true_labels = []
    selected_combos = []
    reward_trace = []
    lagrangian_reward_trace = []
    errors = 0
    remaining_training_budget_per_sample = training_remaining_budget / T2

    combo_reward_sum = {c: 0.0 for c in view_combinations}
    combo_reward_count = {c: 0 for c in view_combinations}

    for t in range(T1, total_samples):

        # ------------------------------------------------------------
        # SAFE BUDGET CHECK (BwK-style absorbing fallback policy)
        # If budget is exhausted, switch to free/zero-cost action only
        # (related to safe action fallback ideas in BwK / constrained bandits)
        # ------------------------------------------------------------
        if training_remaining_budget <= 0:
            # deterministic fallback to cheapest (free) action
            free_idx = int(np.argmin([combo_costs[c] for c in view_combinations]))
            idx = free_idx
            combo = view_combinations[idx]
            cost = combo_costs[combo]
            probs = None  # not used in fallback mode
        else:
            # Compute probabilities over ALL actions: (EXP4: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf, Algorithm 3)
            weight_sum = weights.sum()
            probs = (1 - gamma) * (weights / weight_sum) + gamma / n_experts
            # Sample action
            idx = int(rng.choice(n_experts, p=probs))
            combo = view_combinations[idx]
            cost = combo_costs[combo]
            if cost > training_remaining_budget:
                feasible_idx = np.array([
                    i for i, c in enumerate(view_combinations)
                    if combo_costs[c] <= training_remaining_budget + 1e-12
                ], dtype=int)

                if feasible_idx.size == 0:
                    feasible_idx = np.array([
                        int(np.argmin([combo_costs[c] for c in view_combinations]))
                    ], dtype=int)

                idx = int(rng.choice(feasible_idx))   # uniform random fallback
                combo = view_combinations[idx]
                cost = combo_costs[combo]

        # Predict with chosen combo and observe reward and cost
        y_hat, score = predict_combo(x[t], centers, combo, return_score=True)
        reward = float(y_hat == y[t])
        
        lagrangian_reward = reward - lambda_t * cost

        # Record results
        predictions.append(y_hat)
        scores.append(score)
        true_labels.append(y[t])
        selected_combos.append(combo)
        reward_trace.append(reward)
        lagrangian_reward_trace.append(lagrangian_reward)
        errors += int(y_hat != y[t])

        # ------------------------------------------------------------
        # PARTIAL INFORMATION EXP4 UPDATE: (EXP4: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf, Algorithm 3)
        # ------------------------------------------------------------
        if training_remaining_budget > 0:
            # Partial Feedback
            weights[idx] *= np.exp(eta * (lagrangian_reward / probs[idx]))

            combo_reward_sum[combo] += reward / probs[idx]   # importance-weighted
            combo_reward_count[combo] += 1

            # Normalize for numerical stability
            if weights.max() > 1e8:
                weights /= weights.max()

            # ------------------------------------------------------------
            # DUAL UPDATE (Primal-Dual BwK: https://arxiv.org/pdf/1305.2545, Algorithm 2)
            # ------------------------------------------------------------
            lambda_t = max(0.0,lambda_t + alpha * (cost - remaining_training_budget_per_sample))

        # ------------------------------------------------------------
        # BUDGET UPDATE
        # ------------------------------------------------------------
        training_remaining_budget -= cost
        training_budget_spent += cost

    error_rate = errors / len(reward_trace)
    avg_reward = float(np.mean(reward_trace))
    avg_lagrangian_reward = float(np.mean(lagrangian_reward_trace))

    # ------------------------------------------------------------
    # F1 / AUROC (same convention as gmm_2class_bandit_asymmetric.py /
    # gmm_2class_submodular_asymmetric.py's run_phase1_training). No
    # Hungarian label matching is needed here -- unlike those two scripts'
    # unsupervised-start est_means, initialize_centers() is SUPERVISED
    # (learned_centers[yi] = xi the first time class yi is seen), so
    # predictions are already aligned to the true label indices.
    # -- matches the identity label_map used by
    # two_stage_runner.run_inference_lp_dataset for the same reason.
    # ------------------------------------------------------------
    train_f1 = f1_score(true_labels, predictions, zero_division=0)
    try:
        train_auroc = roc_auc_score(true_labels, scores)
    except ValueError:
        # Only one class present among true_labels[T1:].
        train_auroc = float("nan")

    # NOTE: combo_reward_estimates (a Q-function reward table over the full
    # view_combinations power set, computed from the final learned
    # `centers`) used to be built HERE, at the end of every run_alg call --
    # but nothing on the colgen inference path (run_inference_lp_colgen /
    # run_inference_lp_dataset_colgen) consumes it anymore; they compute
    # the equivalent reward directly from `centers` via lp_reward at
    # Phase-2 time instead. Moved into the OLD exhaustive
    # run_inference_lp/run_inference_lp_dataset (below/in two_stage_runner.py)
    # so those still work standalone for comparison, without run_alg paying
    # for an O(2^(nviews-1))-sized dict build on every call regardless of
    # whether anything downstream reads it.

    return {
        'T2': T2,
        'error_rate': error_rate,
        'avg_reward': avg_reward,
        'avg_lagrangian_reward': avg_lagrangian_reward,
        'train_f1': train_f1,
        'train_auroc': train_auroc,
        'training_budget_spent': training_budget_spent,
        'training_budget_spent_ph2': training_budget_spent - T1,
        'training_remaining_budget': training_remaining_budget,
        'lambda_final': lambda_t,
    }

def run_inference_lp_colgen(
    X_inference, Y_inference,
    learned_centers, true_means,
    per_view_costs, inference_budget, m_modalities,
    rng, noise_var=1.0,
):
    """Column-generation replacement for run_inference_lp -- solves the
    SAME LP as solve_LP via core.lp_colgen.solve_lp_policy_colgen instead
    of enumerating the full `view_combinations` power set and looking up a
    precomputed reward table (this uses the 0.5-factor reward convention,
    see core/lp_colgen.py's `lp_reward` docstring -- NOT the no-0.5 formula
    the exhaustive `run_inference_lp`'s own locally-computed
    `combo_reward_estimates` uses; `run_alg` itself no longer builds any
    such table at all, since nothing on this colgen path needs it).

    Thin wrapper: builds the label_map via match_cluster_labels (the one
    thing genuinely specific to synthetic data with known true_means),
    then delegates the actual LP-solve + physical-sampling logic to
    core.lp_colgen.sample_lp_colgen_policy, shared with
    two_stage_runner.run_inference_lp_dataset_colgen's real-data
    counterpart (identity label_map there instead).

    Does NOT need `view_combinations`/`combo_costs`/`combo_reward_estimates`
    as arguments -- takes `per_view_costs` (the plain per-view cost array,
    same convention as `generate_per_view_costs`'s output) instead, so this
    scales to any m_modalities without ever building a 2^(m_modalities-1)
    list. Phase 1 (`run_alg`) still needs the full `view_combinations` list
    regardless (EXP4 requires one weight per expert) -- this only replaces
    Phase 2's LP.
    """
    label_map = match_cluster_labels(learned_centers, true_means)
    return sample_lp_colgen_policy(
        X_inference=X_inference, Y_inference=Y_inference,
        learned_centers=learned_centers, costs=per_view_costs,
        inference_budget=inference_budget, rng=rng, label_map=label_map,
        predict_fn=predict_single_combination, noise_var=noise_var,
    )


def calculate_two_phase_error(T, T1, ph2_error, init_err):
   ph1_scale = T1 / T
   ph2_scale = (T-T1)/T
   return (ph1_scale * init_err) + (ph2_scale * ph2_error)

def solve_LP(view_combinations, combinations_rewards_estimates, combination_costs, M_MODALITIES, Budget, T):
    r = np.array([np.mean(combinations_rewards_estimates[combo]) for combo in view_combinations])
    c = -r

    A_eq = np.ones((1, len(view_combinations)))
    b_eq = np.array([1.0])

    cost_per_combo = np.array([combination_costs[combo] for combo in view_combinations])
    budget_per_sample = Budget / T  # no phase-1 cost to deduct
    A_ub = cost_per_combo.reshape(1, -1)
    b_ub = np.array([budget_per_sample])

    bounds = [(0, 1)] * len(view_combinations)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    return {combo: result.x[i] for i, combo in enumerate(view_combinations)}
    
def run_inference_lp(
    X_inference, Y_inference,
    learned_centers, true_means,
    view_combinations, combo_costs,
    inference_budget, m_modalities,
    rng, noise_var=1.0,
):
    """Exhaustive (non-colgen) Phase-2 LP inference -- kept for small-nviews
    comparison/debugging against run_inference_lp_colgen. NOT used by
    main() by default anymore.

    combo_reward_estimates is now computed HERE, locally, from
    `learned_centers` (no-0.5 `norm.sf(sqrt(snr))` formula -- see
    core/lp_colgen.py's `lp_reward` docstring for how this compares to the
    colgen path's 0.5-factor convention) -- it used to be built inside
    run_alg and threaded through as an argument, but nothing on the colgen
    path needs it anymore, so run_alg no longer builds it; this function
    computes its own copy instead, keeping it self-contained.
    """
    n = len(X_inference)
    label_map = match_cluster_labels(learned_centers, true_means)

    n_views = learned_centers.shape[1]
    mean_diff_sq = np.square(0.5 * (learned_centers[0, :] - learned_centers[1, :]) / np.sqrt(noise_var))
    combo_reward_estimates = {}
    for c in view_combinations:
        mask = np.zeros(n_views, dtype=bool)
        mask[np.array(c) - 1] = True
        snr = np.sum(mean_diff_sq[mask])
        err_rate = norm.sf(np.sqrt(snr))
        combo_reward_estimates[c] = 1 - err_rate

    lp_weights = solve_LP(
        view_combinations=view_combinations,
        combinations_rewards_estimates=combo_reward_estimates,
        combination_costs=combo_costs,
        M_MODALITIES=m_modalities,
        Budget=inference_budget,
        T=n,
    )

    # Extract the probabilities for each combination
    combos_list = list(view_combinations)
    probabilities = np.array([lp_weights[combo] for combo in combos_list])
    
    # Normalize slightly to fix any tiny floating-point inaccuracies from the solver
    probabilities = np.clip(probabilities, 0, 1)
    probabilities /= np.sum(probabilities)

    # ── Step 3: Physically sample the policy for each independent event ──
    actual_inference_rewards = []
    actual_inference_cost = 0.0
    remaining_inference_budget = inference_budget

    # Same sign convention as run_alg's AUROC computation. label_map maps
    # raw predicted cluster index -> true label; since K=2, it's either
    # identity or a full swap, so the score needs the same flip.
    sign_factor = 1.0 if label_map.get(0, 0) == 0 else -1.0

    y_pred_mapped = np.zeros(n, dtype=int)
    evidence_for_class1 = np.zeros(n)

    for i in range(n):
        # Sample a subset S proportionally to the LP policy distribution
        sampled_idx = rng.choice(len(combos_list), p=probabilities)
        sampled_combo = combos_list[sampled_idx]
        
        # Check if we have enough budget left to activate this subset
        cost_of_combo = combo_costs[sampled_combo]
        if remaining_inference_budget - cost_of_combo < 0:
            # Fallback if budget runs out early due to sampling variance
            sampled_combo = (1,)  
            cost_of_combo = combo_costs[sampled_combo]
        else:
            remaining_inference_budget -= cost_of_combo
            actual_inference_cost += cost_of_combo

        # Make the prediction using only the sampled sensors
        raw_pred, score = predict_single_combination(X_inference[i], learned_centers, sampled_combo, return_score=True)
        matched_pred = label_map[raw_pred]
        
        # Record the empirical reward
        is_correct = int(matched_pred == Y_inference[i])
        actual_inference_rewards.append(is_correct)
        y_pred_mapped[i] = matched_pred
        evidence_for_class1[i] = sign_factor * score

    # Empirical mean performance from physical sampling
    inference_accuracy = float(np.mean(actual_inference_rewards))
    inference_error = float(1 - inference_accuracy)

    inference_f1 = f1_score(Y_inference, y_pred_mapped, zero_division=0)
    try:
        inference_auroc = roc_auc_score(Y_inference, evidence_for_class1)
    except ValueError:
        inference_auroc = float("nan")

    return {
        "lp_weights": lp_weights,
        "inference_accuracy": inference_accuracy,
        "inference_error": inference_error,
        "inference_f1": inference_f1,
        "inference_auroc": inference_auroc,
        "actual_cost": actual_inference_cost,
    }

def _plot_metric(df, metric, ylabel, color, title, out_path):

    from math import ceil

    n_trials = df['trial'].nunique()
    budget_fracs = sorted(df['budget_fraction'].unique())
    n_budgets = len(budget_fracs)

    n_cols = min(3, n_budgets)
    n_rows = int(ceil(n_budgets / n_cols))

    # ------------------------------------------------------------
    # GLOBAL Y-AXIS RANGE (mean ± std over all budgets)
    # ------------------------------------------------------------
    agg_all = (
        df.groupby('init_fraction')[metric]
          .agg(['mean', 'std'])
          .reset_index()
    )

    agg_all['std'] = agg_all['std'].fillna(0)

    global_ymin = (agg_all['mean'] - agg_all['std']).min()
    global_ymax = (agg_all['mean'] + agg_all['std']).max()

    pad = (global_ymax - global_ymin) * 0.08 if global_ymax > global_ymin else 0.05
    global_ymin -= pad
    global_ymax += pad

    # ------------------------------------------------------------
    # PLOT SETUP
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
        sharey=True
    )

    axes = np.array(axes).reshape(-1)

    # ------------------------------------------------------------
    # PER BUDGET FRACTION PLOT
    # ------------------------------------------------------------
    for ax_idx, budget in enumerate(budget_fracs):

        ax = axes[ax_idx]
        sub = df[df['budget_fraction'] == budget]

        agg = (
            sub.groupby('init_fraction')[metric]
               .agg(['mean', 'std'])
               .reset_index()
        )

        gamma = agg['init_fraction'].values
        mu = agg['mean'].values
        std = agg['std'].fillna(0).values

        ax.plot(
            gamma, mu,
            marker='o',
            color=color,
            linewidth=1.6,
            markersize=4
        )

        ax.fill_between(
            gamma,
            mu - std,
            mu + std,
            alpha=0.15,
            color=color
        )

        ax.errorbar(
            gamma, mu,
            yerr=std,
            fmt='none',
            ecolor=color,
            elinewidth=0.8,
            capsize=3
        )

        ax.set_ylim(global_ymin, global_ymax)
        ax.set_title(r'$\beta$ (budget fraction) = '+f'{budget}', fontsize=10)
        ax.set_xlabel(r'$\gamma$ (init fraction)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # ------------------------------------------------------------
    # HIDE UNUSED SUBPLOTS
    # ------------------------------------------------------------
    for ax_idx in range(n_budgets, len(axes)):
        axes[ax_idx].set_visible(False)

    # ------------------------------------------------------------
    # FINAL LAYOUT
    # ------------------------------------------------------------
    fig.suptitle(
        f'{title} - mean ± std ({n_trials} trials)',
        fontsize=12,
        y=1.01
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {out_path}")

def plot_alg_results(csv_path):

    df = pd.read_csv(csv_path)
    base = csv_path.replace('.csv', '')

    _plot_metric(
        df,
        metric='error_rate',
        ylabel='Error Rate',
        color='#1d9e75',
        title='Hedge BwK (Full Feedback) Phase-2 Error Rate vs $\\gamma$',
        out_path=f'{base}_error_rate.png',
    )
 
    _plot_metric(
        df,
        metric='two_phase_error',
        ylabel='Two-Phase Error Rate',
        color='#d85a30',
        title='Hedge BwK (Full Feedback) Overall Two-Phase Error Rate vs $\\gamma$',
        out_path=f'{base}_two_phase_error.png'
    ) 

def save_results_to_excel(all_results, excel_path='results_two_stage_asymmetric.xlsx'):
    df = pd.DataFrame(all_results)

    # --- Sheet 1: Detailed results (all trials) ---
    detailed_df = df.copy()

    # --- Sheet 2: Averaged over trials per BUDGET_FRACTION ---
    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns.tolist()
    non_avg_cols = ['trial', 'seed']
    avg_cols = [c for c in numeric_cols if c not in non_avg_cols and c != 'budget_fraction']

    averaged_df = (
        df.groupby(['budget_fraction', 'init_fraction'])[avg_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # Flatten MultiIndex columns
    averaged_df.columns = [
        col[0] if col[1] == '' else f'{col[0]}_{col[1]}'
        for col in averaged_df.columns
    ]

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        averaged_df.to_excel(writer, sheet_name='Averaged', index=False)

    print(f'Results saved to {excel_path}')

def main():
    SEED = 42
    n_trials = 100
    n_samples = 1000
    m_modalities = 5
    k_clusters = 2
    p_y = [0.5] * k_clusters
    cost_full_modalities = 1
    BUDGET_FRACTIONS = np.round(np.arange(0.1, 1.00, 0.1), 2).tolist()
    view_combinations = generate_view_combinations(m_modalities)
    rng = np.random.default_rng(seed=SEED)

    all_results = []

    for trial in range(n_trials):
        seed = SEED + trial
        print(f'Trial {trial + 1}/{n_trials}')
        per_view_costs = generate_per_view_costs(m_modalities, rng)
        combo_costs = generate_combination_costs_heterogeneous(view_combinations, per_view_costs)

        x, y, true_means, _, rng = generate_synthetic_data(rng, n_samples, k_clusters, m_modalities, np.array(p_y), seed)
       
        n_train, n_test = 800, 200
        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]

        for budget_fraction in BUDGET_FRACTIONS:
            total_budget = n_samples * cost_full_modalities * budget_fraction
            training_budget = 0.8 * total_budget
            inference_budget = 0.2 * total_budget
            n_runs = 10
            init_fractions = np.linspace(0, budget_fraction, n_runs).tolist()

            for init_fraction in init_fractions:
                if init_fraction >= budget_fraction:
                    continue

                n_init_samples = int(n_train * init_fraction)
                centers, init_error = initialize_centers(x=x_train, y=y_train, n_init_samples=n_init_samples, k_clusters=k_clusters, m_modalities=m_modalities, rng=rng)
                ph2_result = run_alg(x=x_train, y=y_train, centers=centers, view_combinations=view_combinations, combo_costs=combo_costs, T1=n_init_samples, training_budget=training_budget, rng=rng)
                two_phase_error = calculate_two_phase_error(T=n_train, T1=n_init_samples, ph2_error=ph2_result['error_rate'], init_err=init_error)
                inference_result = run_inference_lp_colgen(X_inference=x_test, Y_inference=y_test, learned_centers=centers, true_means=true_means, per_view_costs=per_view_costs, inference_budget=inference_budget, m_modalities=m_modalities, rng=rng)
                
                all_results.append({
                    'trial': trial,
                    'seed': seed,
                    'budget_fraction': budget_fraction,
                    'init_fraction': init_fraction,
                    'Experts': len(view_combinations),
                    'T': n_train,
                    'T1': n_init_samples,
                    'T2': ph2_result['T2'],
                    'total_budget': total_budget,
                    'initialization_budget': n_init_samples * cost_full_modalities,
                    'training_budget_spent': ph2_result['training_budget_spent'],
                    'training_remaining_budget': ph2_result['training_remaining_budget'],
                    'error_rate': ph2_result['error_rate'],
                    'avg_reward': ph2_result['avg_reward'],
                    'avg_lagrangian_reward': ph2_result['avg_lagrangian_reward'],
                    'two_phase_error': two_phase_error,
                    'inference_length': len(x_test),
                    'inference_budget': inference_budget,
                    'inference_accuracy': inference_result['inference_accuracy'],
                    'inference_error': inference_result['inference_error'],
                    'inference_actual_cost': inference_result['actual_cost'],
                })

    save_results_to_excel(all_results)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Execution time: {end - start:.1f} seconds')