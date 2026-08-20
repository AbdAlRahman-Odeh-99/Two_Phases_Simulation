# -*- coding: utf-8 -*-
"""
adaptive.py
"""

import numba
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from core.two_stage_utils import generate_view_combinations
from core.arm_elimination import arm_elimination_checkpoints, restrict_candidates, eliminate_arms_ucb_lcb
from core.submodular_greedy import (
    ACQUISITION_MODES,             # noqa: F401 -- re-exported
    ARGMAX_ACQUISITION_MODES,      # noqa: F401 -- re-exported
    HEDGE_ACQUISITION_MODES,       # noqa: F401 -- re-exported
    REWARD_ESTIMATES,              # noqa: F401 -- re-exported
    validate_reward_estimate,
    FULL_ENUMERATION_MODES,        # noqa: F401 -- re-exported
    LP_ACQUISITION_MODES,          # noqa: F401 -- re-exported
    MAX_REWARD_ESTIMATE_VIEWS,     # noqa: F401 -- re-exported
    ORACLE_ACQUISITION_MODES,      # noqa: F401 -- re-exported
    REWARD_UPDATE_SCOPES,          # noqa: F401 -- re-exported
    UCB_STRUCTURES,                # noqa: F401 -- re-exported
    pairwise_diff_sq_from_means,
    arm_accuracies_from_means,
    argmax_policy_over_estimates,
    build_arm_tables,
    greedy_chain,
    greedy_oracle,
    lp_policy_over_estimates,
    linprog_policy_over_estimates,
    mask_to_bits,
    structure_ucb_estimates,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
    validate_ucb_structure,
)

from core.logging_utils import get_logger

_log = get_logger("afa.adaptive")

# ─────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────
@numba.njit
def pred_linear_cla(x_observe, class_means):
    mean_tr = class_means.T  # (v, nc)
    diff_mean_sq = mean_tr[:, :, None] - mean_tr[:, None, :]  # (sel, nc, nc)
    pairwise_mean_avg = 0.5 * (mean_tr[:, :, None] + mean_tr[:, None, :])
    inner_prod = np.sum(
        (x_observe[:, None, None] - pairwise_mean_avg) * diff_mean_sq,
        axis=0) > 0  # bool (nc, nc)
    np.fill_diagonal(inner_prod, False)
    y_pred = np.argmax(np.sum(inner_prod, axis=1))
    return y_pred

def class_posterior_scores(x_observe, class_means_sub):
    """softmax(-0.5 * ||x_obs - mu_k[obs]||^2) over classes k -- the exact
    class posterior under equal priors and unit shared variance, restricted
    to the observed views. Used ONLY for AUROC (a continuous per-class
    score); the hard prediction stays pred_linear_cla's for verbatim parity
    with the notebook. Returns shape (nclasses,), rows sum to 1."""
    d2 = np.sum(np.square(x_observe[None, :] - class_means_sub), axis=1)
    logits = -0.5 * d2
    logits -= logits.max()  # numerical stability
    p = np.exp(logits)
    return p / p.sum()


def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _macro_ovr_auroc(y_true, score_mat, nclasses):
    """Macro one-vs-rest AUROC from an (n, nclasses) posterior-score
    matrix; falls back to the standard binary AUROC (positive-class score
    column) when nclasses == 2, and to NaN when a class is missing from
    y_true (roc_auc_score raises ValueError in that case)."""
    try:
        if nclasses == 2:
            return roc_auc_score(y_true, score_mat[:, 1])
        return roc_auc_score(y_true, score_mat, multi_class="ovr",
                             average="macro", labels=np.arange(nclasses))
    except ValueError:
        return float("nan")


def replay_combo_rewards(X_rows, Y_rows, est_means, combo_masks):
    n_arms = combo_masks.shape[0]
    nclasses = est_means.shape[0]
    if len(X_rows) == 0:
        return (np.full(n_arms, 1.0 / nclasses),
                np.ones(n_arms, dtype=np.float64))
    xs = np.asarray(X_rows, dtype=np.float64)
    ys = np.asarray(Y_rows, dtype=int)
    r_hat = np.empty(n_arms, dtype=np.float64)
    for j in range(n_arms):
        m = combo_masks[j]
        d = ((xs[:, None, m] - est_means[None, :, m]) ** 2).sum(axis=2)  # (n, nc)
        r_hat[j] = float(np.mean(d.argmin(axis=1) == ys))
    return r_hat, np.full(n_arms, float(len(xs)), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────
# Training phase: adaptive online training under a training budget
# ─────────────────────────────────────────────────────────────────────────
def run_training_phase(nviews, nclasses, costs, n_train, training_budget,
                        X_train, Y_train, est_means_init, feedback="full",
                        alpha_ucb=2.0, lr=1e-2, step_size=1.0,
                        lambda_max=10.0, rng=None,
                        acquisition="greedy", ucb_structure="flat",
                        reward_update="subsets", arm_elimination=False,
                        force_free=True, true_means=None,
                        reward_estimate="surrogate"):
    """
    Multiclass counterpart of gmm_2class_submodular_asymmetric.
    run_training_phase, wrapping the notebook's `simulation`
    (feedback="bandit") / `simulation_full_feedback` (feedback="full")
    update rules in the repo's phase structure. Returns the same dict
    shape as the binary version (est_means, est_counts, train_reward,
    cum_train_reward, train_f1, train_auroc, label_mapping, spent), with
    label_mapping always the identity (see module docstring, item 3), plus
    a few acquisition-mode diagnostics that callers may ignore.

    feedback : {"full", "bandit"}
        est_means update rule. Unchanged.
    acquisition : {"greedy", "lp_chain", "lp_full", "lp_full_opt",
                   "ucb_argmax", "hedge"}, default "greedy"
        Per-round subset selection. See the module docstring's ACQUISITION
        MODES section. "greedy" reproduces the previous behaviour exactly.
        "ucb_argmax" is the notebook rule: full enumeration + empirical
        UCB + OMD dual, but a deterministic argmax instead of the LP.
    reward_estimate : {"surrogate", "empirical"}, default "surrogate"
        How greedy_oracle / greedy_chain value a subset. Validated for
        MEMBERSHIP, not only for compatibility with `acquisition`: a
        misspelling would otherwise fall through the
        `reward_estimate == "empirical"` test and silently run the
        SURROGATE path under an empirical label.
    true_means : (nclasses, nviews) array, REQUIRED for "lp_full_opt"
        The generative class means. Ignored by every other mode. Supply
        core.optimal_static.synthetic_true_means(...) at the POST-truncation
        view width (pass X_train.shape[1] as n_views_used).
    reward_update : {"subsets", "selected"}, default "subsets"
    Controls empirical arm-reward updates for lp_chain / lp_full and for
    acquisition="greedy" when reward_estimate="empirical".
    Ignored for surrogate greedy and for lp_full_opt.
        "subsets" replays every arm contained in the played subset;
        "selected" scores only the played arm.
    force_free : bool, default True
        Passed to greedy_oracle / greedy_chain -- keeps the free view(s) in
        every acquired subset, matching the LP's and EXP4's invariant.

    rng: warmup rounds' uniform-random predictions (notebook behavior) and,
        under the LP modes, the per-round two-point mixture draw; a fresh
        default_rng(0) is created if None. NOTE: acquisition="greedy"
        consumes rng EXACTLY as before, so old seeds reproduce old results.
    """
    if feedback not in ("full", "bandit"):
        raise ValueError(f"feedback must be 'full' or 'bandit', got {feedback!r}")
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                         f"got {acquisition!r}")
    validate_ucb_structure(ucb_structure)
    if acquisition != "ucb_argmax" and ucb_structure != "flat":
        raise ValueError(
            "ucb_structure is only used by acquisition='ucb_argmax'; "
            f"got acquisition={acquisition!r}, ucb_structure={ucb_structure!r}")
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")
    # MEMBERSHIP check -- see the parameter docstring. Deliberately separate
    # from the acquisition-compatibility check further down, which only
    # fires for the exact string "empirical".
    validate_reward_estimate(reward_estimate)
    # Local flag renamed too: `unbiased` here meant reward_estimate, but
    # sat three lines from ucb_argmax, whose notebook origin is also called
    # "unbiased". `empirical_est` can only mean the one thing.
    empirical_est = (reward_estimate == "empirical" and acquisition in ("greedy", "lp_chain"))
    uses_empirical_arm_rewards = _uses_empirical_arm_rewards(acquisition, reward_estimate)
    # ucb_argmax carries the empirical table unconditionally (like lp_full),
    # so reward_estimate is inert for it -- see the validation below.
    is_argmax = acquisition in ARGMAX_ACQUISITION_MODES
    is_hedge = acquisition in HEDGE_ACQUISITION_MODES
    uses_full_empirical_table = (is_argmax or is_hedge or acquisition == "lp_full")

    if (feedback == "bandit" and reward_update == "subsets" and uses_empirical_arm_rewards):
        raise ValueError(
            "feedback='bandit' with reward_update='subsets' is incoherent: the "
            "counterfactual replay reads y_true, which bandit feedback does not "
            "reveal. Use reward_update='selected' with feedback='bandit', or "
            "feedback='full' with reward_update='subsets'.")
    if acquisition in FULL_ENUMERATION_MODES and nviews > MAX_REWARD_ESTIMATE_VIEWS:
        raise ValueError(
            f"acquisition={acquisition!r} enumerates 2^(nviews-1) = 2^{nviews - 1} "
            f"subsets eagerly; nviews={nviews} exceeds "
            f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. Use "
            f"acquisition='lp_chain' (nviews+1 arms), trim with "
            f"max_modalities, or use acquisition='greedy'.")

    is_oracle = acquisition in ORACLE_ACQUISITION_MODES
    if is_oracle:
        if true_means is None:
            raise ValueError(
                f"acquisition={acquisition!r} needs the TRUE generative means, "
                f"which exist only for the synthetic datasets. Pass "
                f"true_means=core.optimal_static.synthetic_true_means(...), or "
                f"use acquisition='lp_full' for the learned-estimate version of "
                f"the same action space.")
        true_means = np.asarray(true_means, dtype=np.float64)
        if true_means.shape != (nclasses, nviews):
            raise ValueError(
                f"true_means must have shape ({nclasses}, {nviews}), got "
                f"{true_means.shape}. synthetic_true_means must be called with "
                f"n_views_used=X_train.shape[1] -- the generator's width and "
                f"the post-max_modalities width are not the same thing.")
    if rng is None:
        rng = np.random.default_rng(0)

    est_means = np.asarray(est_means_init, dtype=np.float64).copy()
    if est_means.shape != (nclasses, nviews):
        raise ValueError(
            f"est_means_init must have shape ({nclasses}, {nviews}), got {est_means.shape}"
        )

    est_counts = np.ones((nclasses, nviews))
    one_vec = np.ones(nclasses)

    free_indices = [idx for idx in range(nviews) if costs[idx] == 0]
    free_only_subset = np.zeros(nviews, dtype=bool)
    free_only_subset[free_indices] = True

    remaining_budget = training_budget
    spending_ratio = training_budget / n_train
    omd_lambda = 0.0

    record_pred = np.zeros(n_train, dtype=int)
    record_scores = np.zeros((n_train, nclasses))
    total_spent = 0.0

    # LP-over-arms state
    combo_masks = combo_cost = cost_order = arm_bits = None
    bit_index = r_hat = combo_counts = None



    if reward_estimate == "empirical" and acquisition not in ("greedy", "lp_chain"):
        raise ValueError(
            f"reward_estimate='empirical' applies to acquisition='greedy' or "
            f"'lp_chain' (the two modes that consume the Bhattacharyya "
            f"surrogate); got acquisition='{acquisition}'. lp_full and "
            f"ucb_argmax/hedge already score arms from the empirical accuracy "
            f"table unconditionally, so the flag has nothing to switch "
            f"there -- leave it at 'surrogate' and it is simply inert.")
    # Arm-elimination compatibility.
    if arm_elimination:
        if acquisition == "lp_full_opt":
            raise ValueError(
                "arm_elimination is not supported with "
                "acquisition='lp_full_opt'. Its oracle distribution is "
                "computed once and does not use the active-arm mask."
            )
        if acquisition == "greedy" and not empirical_est:
            raise ValueError(
                "Adaptive arm elimination with acquisition='greedy' "
                "requires reward_estimate='empirical'. Surrogate greedy "
                "does not maintain an enumerated empirical arm table."
            )
        if acquisition == "lp_chain" and not empirical_est:
            raise ValueError(
                "Adaptive arm elimination with acquisition='lp_chain' "
                "requires reward_estimate='empirical'. Surrogate lp_chain "
                "rebuilds its arm table as the estimated means change, so "
                "a persistent active-arm mask is not supported."
            )
    if empirical_est or uses_full_empirical_table:
        if nviews > MAX_REWARD_ESTIMATE_VIEWS:
            raise ValueError(
                f"reward_estimate='empirical' enumerates 2^(nviews-1) = "
                f"2^{nviews - 1} arms; nviews={nviews} exceeds "
                f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. "
                f"Use reward_estimate='surrogate' (the Bhattacharyya proxy), "
                f"which needs no enumeration.")
        tables = build_arm_tables(generate_view_combinations(nviews), costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        r_hat = np.full(len(arm_bits), 1.0 / nclasses, dtype=np.float64)
        combo_counts = np.ones(len(arm_bits), dtype=np.float64)

    # Persistent statistics indexed by the subset's bitmask.
    arm_reward_stats = {}
    arm_count_stats = {}

    b_allowance = spending_ratio
    p_oracle = None
    warmup_rows = []
    views_trace = np.zeros(n_train)
    seen_masks = set()
    selected_subsets = [None] * n_train



    # ORACLE SETUP (acquisition="lp_full_opt" only)
    if is_oracle:
        tables = build_arm_tables(generate_view_combinations(nviews), costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        r_hat = arm_accuracies_from_means(X_train, Y_train, true_means, combo_masks)
        combo_counts = np.full(len(arm_bits), float(n_train))
        p_oracle, omd_lambda = linprog_policy_over_estimates(r_hat, combo_cost, spending_ratio)

    # ONE-TIME HEDGE SETUP (acquisition="hedge" only)
    if is_hedge:
        hedge_v = np.ones(2, dtype=np.float64)
        hedge_epsilon = np.sqrt(np.log(2.0) / n_train)

    elimination_trace = []
    if arm_elimination and combo_masks is not None:
        active_arms = np.ones(len(combo_masks), dtype=bool)
        initial_arm_count = int(active_arms.sum())
        elimination_points = set(arm_elimination_checkpoints(n_train))
    else:
        active_arms = None
        initial_arm_count = 0
        elimination_points = set()

    for t in range(n_train):
        if t < nclasses:
            subset = np.ones(nviews, dtype=bool)
            is_init = True
        elif acquisition == "greedy":
            #remaining_rounds = n_train - t
            #full_cost = float(np.sum(costs))
            ## Temporary high-budget diagnostic:
            ## force every view when all remaining rounds can be fully observed.
            #if (remaining_budget + 1e-12 >= remaining_rounds * full_cost):
            #    subset = np.ones(nviews, dtype=bool)
            if empirical_est:
                # Empirical-table branch (Greedy)
                round_idx = max(t - nclasses, 0)
                log_bonus = (alpha_ucb * np.log(round_idx + 2))
                # Before the first successful elimination, all arms are active,
                # so preserve the original unconstrained Greedy path.
                if (active_arms is not None and not np.all(active_arms)):
                    greedy_active_bits = arm_bits[active_arms]
                else:
                    greedy_active_bits = None
                def gain_func(sel):
                    m = np.zeros(nviews, dtype=bool)
                    m[np.asarray(sel, dtype=int)] = True
                    j_global = bit_index[mask_to_bits(m)]
                    return float(r_hat[j_global] + np.sqrt(log_bonus / combo_counts[j_global]))
                    
                subset = greedy_oracle(None, costs, omd_lambda,
                                       remaining_budget, free_indices,
                                       force_free=force_free,
                                       gain_func=gain_func,
                                       empty_value=1.0 / nclasses,
                                       active_arm_bits=greedy_active_bits)
            else:
                # Biased branch (Greedy)
                diff_mean_sq = pairwise_diff_sq_from_means(est_means)  # (nv, nc, nc)
                inv_sqrt_cnt = np.sqrt(1.0 / est_counts).T  # (nv, nc)
                bonus_mat = inv_sqrt_cnt[:, :, None] + inv_sqrt_cnt[:, None, :]
                bonus_mat *= np.sqrt(alpha_ucb * np.log(t + 1))
                optimistic_diff_mean_sq = diff_mean_sq + bonus_mat
                subset = greedy_oracle(optimistic_diff_mean_sq, costs, omd_lambda, remaining_budget, free_indices, force_free=force_free)
            is_init = False
        elif is_argmax:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                round_idx = max(t - nclasses, 0)
                # Structure the full permanent arm table before restricting
                # candidates, so an inactive or unaffordable arm can still
                # inform an eligible arm through the subset Hasse diagram.
                raw_ucb = (r_hat + np.sqrt(
                    alpha_ucb * np.log(round_idx + 2) / combo_counts))
                structured_ucb = structure_ucb_estimates(
                    raw_ucb, arm_bits, bit_index, ucb_structure)
                # GLOBAL arm-table indices.
                affordable_idx = np.flatnonzero(combo_cost <= remaining_budget + 1e-12)
                candidate_idx = restrict_candidates(affordable_idx, active_arms,)
                candidate_cost = combo_cost[candidate_idx]
                candidate_ucb = structured_ucb[candidate_idx]
                # j_local indexes candidate_ucb/candidate_cost.
                j_local = argmax_policy_over_estimates(candidate_ucb, candidate_cost, omd_lambda, remaining_budget,)
                # Convert the local result back to the permanent arm table.
                j_global = int(candidate_idx[j_local])
                subset = combo_masks[j_global].copy()
            is_init = False
        elif is_hedge:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                round_idx = max(t - nclasses, 0)
                # GLOBAL indices of affordable and active arms.
                affordable_idx = np.flatnonzero(combo_cost <= remaining_budget + 1e-12)
                candidate_idx = restrict_candidates(affordable_idx, active_arms,)
                candidate_cost = combo_cost[candidate_idx]
                # Confidence calculation is performed only for candidates.
                candidate_ucb = (r_hat[candidate_idx] + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts[candidate_idx]))                
                hedge_y = hedge_v / hedge_v.sum()
                candidate_effective_cost = (hedge_y[0] + hedge_y[1] * candidate_cost)
                candidate_score = candidate_ucb / candidate_effective_cost
                # j_local indexes candidate_score.
                j_local = int(np.argmax(candidate_score))
                # Convert to the global arm index.
                j_global = int(candidate_idx[j_local])
                # Adaptive result.
                subset = combo_masks[j_global].copy()
            is_init = False
        elif is_oracle:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                subset = combo_masks[int(rng.choice(len(p_oracle), p=p_oracle))].copy()
            is_init = False
        else:
            # ── LP over an ENUMERATED arm set ("lp_chain" / "lp_full")
            rebuild_arms = ((acquisition == "lp_chain" or combo_masks is None)
                            and not empirical_est)
            act_idx = None
            if empirical_est:
                # This can only be empirical lp_chain. Empirical greedy
                # was already handled by the earlier greedy branch.
                round_idx_u = max(t - nclasses, 0)
                log_bonus = alpha_ucb * np.log(round_idx_u + 2)
                def gain_func(sel):
                    m = np.zeros(nviews, dtype=bool)
                    m[np.asarray(sel, dtype=int)] = True
                    j_global = bit_index[mask_to_bits(m)]
                    return float(r_hat[j_global] + np.sqrt(log_bonus / combo_counts[j_global]))
                combos = greedy_chain(est_means, costs, free_indices, force_free=force_free, gain_func=gain_func, empty_value=1.0 / nclasses,)
                act_idx = np.array([bit_index[sum(1 << (v - 1) for v in combo)] for combo in combos], dtype=int,)

            if rebuild_arms:
                # Biased branch (LP approaches)
                if acquisition == "lp_chain":
                    combos = greedy_chain(est_means, costs, free_indices, force_free=force_free,)
                else:
                    combos = generate_view_combinations(nviews)

                tables = build_arm_tables(combos, costs, nviews)
                combo_masks = tables["combo_masks"]
                combo_cost = tables["combo_cost"]
                cost_order = tables["cost_order"]
                arm_bits = tables["arm_bits"]
                bit_index = tables["bit_index"]

                seed_rewards, seed_counts = replay_combo_rewards(X_train[warmup_rows], Y_train[warmup_rows], est_means, combo_masks)
                r_hat = np.empty(len(arm_bits), dtype=np.float64)
                combo_counts = np.empty(len(arm_bits), dtype=np.float64)
                for j, bits in enumerate(arm_bits):
                    key = int(bits)
                    if key not in arm_reward_stats:
                        arm_reward_stats[key] = float(seed_rewards[j])
                        arm_count_stats[key] = float(seed_counts[j])
                    r_hat[j] = arm_reward_stats[key]
                    combo_counts[j] = arm_count_stats[key]

            # adaptive per-round alloance
            b_allowance = max(0.0, remaining_budget) / max(1, n_train - t)

            if remaining_budget <= 0:
                # Absorbing fallback policy once the pool is gone.
                subset = free_only_subset.copy()
            else:
                round_idx = t - nclasses  # 0-based, drives the log(t+1) bonus
                if empirical_est:
                    # Keep only active arms from the newly constructed chain.
                    candidate_idx = restrict_candidates(act_idx, active_arms,)
                    # Calculate UCB only for those active chain arms.
                    candidate_ucb = (r_hat[candidate_idx] + np.sqrt(log_bonus / combo_counts[candidate_idx]))
                else:
                    # lp_full, or surrogate lp_chain without elimination:
                    # restrict first, then calculate UCB only for candidates.
                    candidate_idx = restrict_candidates(np.arange(len(combo_masks), dtype=int), active_arms,)
                    candidate_ucb = (r_hat[candidate_idx] + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts[candidate_idx]))
                candidate_cost = combo_cost[candidate_idx]
                p_lp, omd_lambda = linprog_policy_over_estimates(candidate_ucb, candidate_cost, b_allowance,)
                # Local index into candidate_ucb/candidate_cost/p_lp.
                j_local = int(rng.choice(len(p_lp), p=p_lp,))
                # Map back to the permanent arm-table index.
                j_global = int(candidate_idx[j_local])
                subset = combo_masks[j_global].copy()
            is_init = False

        # budget check
        inst_cost = np.sum(costs[subset])
        if remaining_budget >= inst_cost:
            remaining_budget -= inst_cost
        else:
            subset = free_only_subset.copy()
            inst_cost = np.sum(costs[subset])  # == 0
            remaining_budget -= inst_cost
        if is_hedge:
            resource_z = np.array([1.0, inst_cost], dtype=np.float64,)
            hedge_v *= ((1.0 + hedge_epsilon)** resource_z)
        total_spent += inst_cost
        views_trace[t] = int(subset.sum())
        seen_masks.add(subset.tobytes())
        selected_subsets[t] = tuple((np.flatnonzero(subset) + 1).tolist())
        if is_init and subset.all():
            warmup_rows.append(t)

        # observe acquired views only (features are semi-bandit in BOTH modes)
        x_obs = X_train[t, subset]
        means_sub = est_means[:, subset]
        if not is_init:
            y_pred = int(pred_linear_cla(x_obs, means_sub))
        else:
            y_pred = int(rng.integers(nclasses))
        record_pred[t] = y_pred
        record_scores[t] = class_posterior_scores(x_obs, means_sub)

        y_true = int(Y_train[t])
        reward = y_pred == y_true

        # per-arm reward-estimate update
        if (combo_masks is not None and not is_oracle and not (empirical_est and is_init)):
            played_bits = mask_to_bits(subset)
            if reward_update == "selected":
                # BANDIT scope
                # j0 = bit_index.get(played_bits)
                # targets = [] if j0 is None else [(j0, float(reward))]
                j0 = bit_index.get(played_bits)
                if (j0 is not None and (active_arms is None or active_arms[j0])):
                    targets = [(j0, float(reward))]
                else:
                    targets = []
            else:
                # COUNTERFACTUAL REPLAY
                targets = []
                candidate_idx = restrict_candidates(np.arange(len(arm_bits), dtype=int), active_arms,)
                candidate_bits = arm_bits[candidate_idx]
                contained_local = ((candidate_bits & played_bits) == candidate_bits)
                contained_idx = candidate_idx[contained_local]
                for k in contained_idx:
                    m_k = combo_masks[k]
                    y_sub = int(pred_linear_cla(X_train[t, m_k], est_means[:, m_k]))
                    targets.append((int(k), float(y_sub == y_true)))

            for j0, r_obs in targets:
                combo_counts[j0] += 1.0
                r_hat[j0] += (r_obs - r_hat[j0]) / combo_counts[j0]
                # Preserve statistics under the subset identity.
                key = int(arm_bits[j0])
                arm_reward_stats[key] = float(r_hat[j0])
                arm_count_stats[key] = float(combo_counts[j0])

        # Update means
        if feedback == "full":
            # y_true revealed every round: running mean of the TRUE class
            est_counts[y_true, subset] += 1
            est_means[y_true, subset] += ((1.0 / est_counts[y_true, subset]) * (x_obs - est_means[y_true, subset]))
        else:  # bandit
            est_counts[y_pred, subset] += 1
            if reward:
                # correct prediction: y_pred == y_true, running-mean update
                est_means[y_pred, subset] += ((1.0 / est_counts[y_pred, subset]) * (x_obs - est_means[y_pred, subset]))
            else:
                # incorrect prediction (complementary-label update)
                eliminated = y_pred
                elim = np.zeros(nclasses)
                elim[eliminated] = 1.0
                l_grad = -2 * (x_obs[None, :] - means_sub)  # (nc, ns)
                grad = l_grad * (one_vec - (nclasses - 1) * elim)[:, None]
                est_means[:, subset] -= lr * grad

        # # ONE-TIME EMPIRICAL-TABLE WARMUP SEEDING
        # if empirical_est and t == nclasses - 1:
        #     seed_rewards, seed_counts = replay_combo_rewards(X_train[warmup_rows], Y_train[warmup_rows], est_means, combo_masks,)
        #     r_hat[:] = seed_rewards
        #     combo_counts[:] = seed_counts

        # OMD dual update
        # ucb_argmax joins greedy here and NOT the LP modes: the LP gets its
        # lambda back from the solver as the budget's shadow price, whereas
        # the argmax has no constraint to price and would buy the full view
        # set every round if lambda stayed at 0.
        if acquisition in ("greedy",) + ARGMAX_ACQUISITION_MODES:
            raw_lambda = omd_lambda + step_size * (inst_cost - spending_ratio)
            omd_lambda = max(0, min(lambda_max, raw_lambda))

        completed = t + 1
        if (arm_elimination and active_arms is not None and completed in elimination_points):
            remaining_rounds = n_train - completed
            b_elim = (max(0.0, remaining_budget) / max(1, remaining_rounds))
            active_arms, elim_info = eliminate_arms_ucb_lcb(r_hat=r_hat, combo_counts=combo_counts, combo_cost=combo_cost, active_arms=active_arms, alpha_ucb=alpha_ucb, round_idx=completed, budget_per_round=b_elim,)
            _log.info("[arm elimination] t=%d/%d: %d -> %d active arms (removed %d)", completed, n_train, elim_info["before"], elim_info["after"], elim_info["eliminated"],)
            elimination_trace.append({
                "completed_rounds": int(completed),
                "before": int(elim_info["before"]),
                "after": int(elim_info["after"]),
                "eliminated": int(elim_info["eliminated"]),
                "budget_per_remaining_round": float(b_elim),
            })

    correct_vec = (record_pred == np.asarray(Y_train, dtype=int)).astype(float)
    train_acc = float(np.mean(correct_vec))
    cum_train_acc = np.cumsum(correct_vec) / np.arange(1, n_train + 1)
    train_f1 = _macro_f1(Y_train, record_pred)
    train_auroc = _macro_ovr_auroc(Y_train, record_scores, nclasses)

    return {
        "est_means": est_means,
        "est_counts": est_counts,
        "train_reward": train_acc,
        "cum_train_reward": cum_train_acc,
        "train_f1": train_f1,
        "train_auroc": train_auroc,
        "label_mapping": {k: k for k in range(nclasses)},  # identity -- see docstring
        "spent": total_spent,
        # ── acquisition diagnostics (extra keys; existing callers ignore) ──
        "acquisition": acquisition,
        "ucb_structure": ucb_structure,
        "reward_update": (reward_update if uses_empirical_arm_rewards else ""),
        "n_arms": 0 if combo_masks is None else int(combo_masks.shape[0]),
        "combo_rewards": r_hat,
        "combo_counts": combo_counts,
        "lambda_final": float(omd_lambda),
        # Last value of the now per-round LP allowance (== spending_ratio
        # under greedy, which never touches it, and the flat one-shot
        # allowance under lp_full_opt).
        "b_allowance_final": float(b_allowance),
        "oracle_probs": p_oracle,
        "avg_views_acquired": float(np.mean(views_trace)),
        "n_unique_masks": len(seen_masks),
        "selected_subsets": selected_subsets,
        # Elimination
        "arm_elimination": bool(arm_elimination),
        "initial_arms": int(initial_arm_count),
        "final_active_arms": (int(active_arms.sum()) if active_arms is not None else int(initial_arm_count)),
        "num_eliminated": (int(initial_arm_count - active_arms.sum()) if active_arms is not None else 0),
        "elimination_trace": elimination_trace,
    }


# ─────────────────────────────────────────────────────────────────────────
# Inference phase: LP-based inference policy + physical sampling
# ─────────────────────────────────────────────────────────────────────────
def run_inference_phase(masks, probs, combo_costs, costs, est_means,
                         inference_budget, X_inf, Y_inf, rng):
    n = len(X_inf)
    nclasses = est_means.shape[0]
    remaining_budget = inference_budget
    idxs = np.arange(len(masks))

    fallback_subset = np.zeros(len(costs), dtype=bool)
    fallback_subset[0] = True
    fallback_cost = costs[0]

    correct = 0
    spent = 0.0
    y_pred = np.zeros(n, dtype=int)
    score_mat = np.zeros((n, nclasses))
    for t in range(n):
        sel_i = rng.choice(idxs, p=probs)
        subset = masks[sel_i]
        cost = combo_costs[sel_i]

        if remaining_budget - cost < 0:
            subset = fallback_subset
            cost = fallback_cost

        remaining_budget -= cost
        spent += cost

        x_obs = X_inf[t, subset]
        means_sub = est_means[:, subset]
        pred = int(pred_linear_cla(x_obs, means_sub))
        correct += int(pred == Y_inf[t])
        y_pred[t] = pred
        score_mat[t] = class_posterior_scores(x_obs, means_sub)

    f1 = _macro_f1(Y_inf, y_pred)
    auroc = _macro_ovr_auroc(Y_inf, score_mat, nclasses)

    return {
        "inference_reward": correct / n,
        "spent": spent,
        "inference_f1": f1,
        "inference_auroc": auroc,
    }
