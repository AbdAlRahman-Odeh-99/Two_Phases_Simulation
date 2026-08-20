# -*- coding: utf-8 -*-
"""
two_stage_multiclass_greedy.py

=== acquisition: which subset the policy buys each round ===
two_stage's premise is that the CLASSIFIER is frozen after Stage 1 --
the centres learned there are what Stage 2 predicts with and what
inference predicts with, unchanged. This module keeps that premise under
EVERY mode, so the only axis left is the ACQUISITION policy, and it is
spelled exactly as adaptive spells it (both read
ACQUISITION_MODES from core.submodular_greedy):

  "greedy"    (DEFAULT) The submodular greedy oracle, re-derived each
              round from the frozen centres, the current lambda and the
              remaining budget. Fully static -- it never learns.
  "lp_chain"  UCB reward estimates over the nviews+1 nested greedy chain
              (see greedy_chain), with the per-round budgeted LP solved
              exactly. Runs at any nviews.
  "lp_full"   The same over the FULL 2^(nviews-1) enumeration -- the
              small-nviews fidelity check for what the chain restriction
              costs. Capped at MAX_REWARD_ESTIMATE_VIEWS views.
  "lp_full_opt"
              ORACLE CEILING for lp_full. Same action space, but the arm
              values are not estimated: they are the exact accuracies of the
              TRUE generative means on the Stage-2 rows (caller passes
              true_means, so SYNTHETIC-ONLY), and the LP is solved ONCE
              before the Stage-2 loop. Each round is then a single draw from
              a frozen distribution -- no UCB, no re-solve, no r_hat update.
              lp_full_opt minus lp_full is the price of LEARNING the arm
              values, holding action space and LP fixed.
  "ucb_argmax"
              The NOTEBOOK rule (multiclass_supervised_unbiased_adaptive.
              ipynb's `sim_unbiased`), ported to this module and to
              adaptive identically. Same full 2^(nviews-1)
              arm table and same empirical r_hat as lp_full, but the
              per-round policy is a DETERMINISTIC Lagrangian argmax,

                  argmax_S  r_hat[S] - lambda_t * cost(S) + bonus[S],

              with lambda_t the OMD dual "greedy" runs rather than an LP
              shadow price. So ucb_argmax minus lp_full is the price of
              committing to one arm instead of mixing over a distribution,
              holding action space and reward table fixed -- the natural
              companion to the lp_full_opt minus lp_full comparison above.
              Capped at MAX_REWARD_ESTIMATE_VIEWS views like lp_full. Here
              the r_hat it argmaxes over is seeded from Stage 1, so unlike
              the submodular port it starts from real accuracies on round
              one. NOT capped at 1.0 -- see below.

=== Instrumentation ===
The Stage-2 loop's numbered sections are timed individually into
t_reward_update, t_dual_update and t_predict (t_acquisition is measured
one level down, inside core.submodular_greedy's policy routines, so it is
NOT ticked here -- ticks accumulate per name and a second tick around the
same span would double-count it). Nothing about the loop's control flow,
ordering or arithmetic changed.

`trace_rounds=True` additionally records one dict per Stage-2 round
(subset, cost, lambda, reward, remaining budget) and returns it under
'round_trace'. That is the honest home for the training trace: the
'selected_subsets' list this function already returns is serialised into a
SINGLE Excel cell by the runner, thousands of subsets long, stripped of the
lambda and budget state that would make it interpretable. The trace keeps
the state and lands in a JSONL sidecar. selected_subsets is unchanged and
still returned, so nothing that consumed it breaks.
"""

from __future__ import annotations

import numpy as np

from core.logging_utils import get_logger, tick
from core.two_stage_utils import generate_view_combinations
from core.arm_elimination import arm_elimination_checkpoints, restrict_candidates, eliminate_arms_ucb_lcb
from core.submodular_greedy import (
    # Shared policy vocabulary -- core is the canonical definition, so this
    # method and adaptive agree on what run_proposed_
    # methods.py's --acquisition / --reward-update values mean.
    ACQUISITION_MODES,           # noqa: F401 -- re-exported
    ARGMAX_ACQUISITION_MODES,    # noqa: F401 -- re-exported
    HEDGE_ACQUISITION_MODES,     # noqa: F401 -- re-exported
    REWARD_ESTIMATES,            # noqa: F401 -- re-exported
    validate_reward_estimate,
    FULL_ENUMERATION_MODES,      # noqa: F401 -- re-exported
    LP_ACQUISITION_MODES,        # noqa: F401 -- re-exported
    MAX_REWARD_ESTIMATE_VIEWS,   # noqa: F401 -- re-exported
    ORACLE_ACQUISITION_MODES,    # noqa: F401 -- re-exported
    REWARD_UPDATE_SCOPES,        # noqa: F401 -- re-exported
    pairwise_diff_sq_from_means,
    arm_accuracies_from_means,
    argmax_policy_over_estimates,
    greedy_chain,                # noqa: F401 -- re-exported (moved here)
    greedy_oracle,
    lp_policy_over_estimates,    # noqa: F401 -- re-exported (moved here)
    linprog_policy_over_estimates,
    build_arm_tables,
    mask_to_bits,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,   # noqa: F401
)

from core.multiclass_common import (
    PRED_RULES,                                  # noqa: F401 -- re-exported
    _class_posterior_scores,
    _macro_f1,
    _macro_ovr_auroc,
    _pred_nearest_center,
    _pred_pairwise_vote,
)

LP_ACTION_SPACE = {"lp_chain": "chain", "lp_full": "full", "lp_full_opt": "full"}

_log = get_logger("afa.two_stage")


# Mask-space prediction
def predict_mask_multiclass(x_sample, centers, mask, return_score=False, pred_rule="nearest_center"):
    x_obs = x_sample[mask]
    means_sub = centers[:, mask]
    if pred_rule == "nearest_center":
        pred = _pred_nearest_center(x_obs, means_sub)
    elif pred_rule == "pairwise_vote":
        pred = _pred_pairwise_vote(x_obs, means_sub)
    else:
        raise ValueError(f"pred_rule must be one of {PRED_RULES}, got {pred_rule!r}")
    if return_score:
        return pred, _class_posterior_scores(x_obs, means_sub)
    return pred

def stage1_counts(y, T1, nclasses, nviews):
    per_class = 1 + np.bincount(np.asarray(y[:T1], dtype=int), minlength=nclasses)
    return np.repeat(per_class[:nclasses, None], nviews, axis=1).astype(np.float64)

def stage1_combo_rewards(x, y, centers, T1, combos, nviews, pred_rule="nearest_center",):
    """
    Replay the frozen Stage-1 classifier over every combination using the
    same prediction rule used during Stage 2.
    """
    if T1 <= 0:
        return (np.full(len(combos), 1.0 / centers.shape[0], dtype=np.float64,), np.ones(len(combos), dtype=np.float64),)

    xs = np.asarray(x[:T1], dtype=np.float64)
    ys = np.asarray(y[:T1], dtype=int)
    r_hat = np.empty(len(combos), dtype=np.float64)
    for j, combo in enumerate(combos):
        mask = np.zeros(nviews, dtype=bool)
        mask[np.asarray(combo, dtype=int) - 1] = True
        predictions = np.array([predict_mask_multiclass(x_sample, centers, mask, pred_rule=pred_rule,) for x_sample in xs], dtype=int,)
        r_hat[j] = float(np.mean(predictions == ys))
    combo_counts = np.full(len(combos), float(T1), dtype=np.float64,)
    return r_hat, combo_counts

# Stage 2 (training phase)
def run_alg_greedy_multiclass(x, y, centers, costs, T1, training_budget, rng,
                              acquisition="greedy", alpha_ucb=2.0,
                              step_size=1.0, lambda_max=10.0,
                              pred_rule="nearest_center",
                              force_free=True, reward_update="subsets",
                              true_means=None, reward_estimate="surrogate", arm_elimination=False,
                              trace_rounds=False):
    """Greedy counterpart of two_stage_multiclass.run_alg_multiclass.

    Parameters
    ----------
    x, y : (n, nviews), (n,)
        The FULL training arrays. Rounds T1..n-1 are Stage 2 (rounds
        0..T1-1 were consumed by Stage 1), exactly as in the EXP4 version.
    centers : (nclasses, nviews)
        Stage-1 learned centres. NEVER mutated in place, and never updated:
        an internal copy is returned under the 'centers' key so the caller
        can hand the same object to inference without special-casing.
    costs : (nviews,) array
        Per-view costs, 0-indexed, free views at cost 0. This REPLACES the
        EXP4 version's (view_combinations, combo_costs) pair.
    T1, training_budget, rng, step_size, lambda_max, pred_rule
        Identical meaning to run_alg_multiclass.
    acquisition : {"greedy", "lp_chain", "lp_full", "lp_full_opt",
                   "ucb_argmax", "hedge"}
        The per-round acquisition policy -- see the module docstring.
        Default "greedy". Same vocabulary as
        adaptive.run_training_phase's flag of the same
        name (both validate against core.submodular_greedy's
        ACQUISITION_MODES), which is what makes the two methods comparable
        on this axis.
    reward_update : {"subsets", "selected"}
        Arm-scoring scope for lp_chain / lp_full / ucb_argmax / hedge; inert under
        "greedy", which has no enumerated arms, and under "lp_full_opt",
        whose arm values are exact and never scored.
    reward_estimate : {"surrogate", "empirical"}
        How greedy_oracle / greedy_chain value a subset. Validated for
        MEMBERSHIP here, not only for compatibility with `acquisition`: a
        misspelling used to fall through the `reward_estimate ==
        "empirical"` test and silently run the SURROGATE path under an
        empirical label.
    true_means : (nclasses, nviews) array, REQUIRED for "lp_full_opt"
        The generative class means -- SYNTHETIC datasets only. Supply
        core.optimal_static.synthetic_true_means(...) at the POST-truncation
        view width. NOTE this does not touch the classifier: Stage 2 still
        predicts with the frozen Stage-1 `centers`, so the mode isolates the
        acquisition axis and its accuracy is NOT an upper bound on what a
        method with oracle CENTRES could reach.
    alpha_ucb : float, default 2.0
        Optimism scale in the bonus sqrt(alpha_ucb * log(t+1)) / sqrt(count)
        -- same knob and same default as
        adaptive.run_training_phase. alpha_ucb=0 makes
        the oracle purely exploitative (and, under acquisition="greedy",
        fully deterministic given lambda). The resulting UCB is capped at
        1.0 -- see the module docstring's UCB CAPPING section.
    force_free : bool, default True
        Passed through to greedy_oracle -- keeps the free view(s) in every
        acquired subset, matching EXP4's and the LP's invariant.
    trace_rounds : bool, default False
        Record one dict per Stage-2 round under the returned 'round_trace'
        key: round index, acquired subset, its cost, lambda, the 0/1
        reward, and the budget left afterwards. OFF by default -- it is
        O(T2) dicts per sweep cell, which is the same order as the
        selected_subsets list already built, but there is no reason to pay
        it on runs that will not look at it.

    Returns
    -------
    dict with the SAME keys run_alg_multiclass returns (so the runner and
    run_proposed_methods.normalize_two_stage_mc need no changes), plus
    'centers', 'est_counts', 'acquisition', 'avg_views_acquired',
    'n_unique_masks', 'round_trace', 'n_budget_fallbacks'. 'warm_start' is
    always False -- there is no expert weight vector to warm-start here;
    the Stage-1 information enters through the centres and (via
    stage1_counts) the optimism bonus instead.
    """
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                         f"got {acquisition!r}")
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")

    validate_reward_estimate(reward_estimate)
    use_lp = acquisition in LP_ACTION_SPACE
    is_oracle = acquisition in ORACLE_ACQUISITION_MODES
    is_argmax = acquisition in ARGMAX_ACQUISITION_MODES
    is_hedge = acquisition in HEDGE_ACQUISITION_MODES
    uses_full_empirical_table = (is_argmax or is_hedge or acquisition == "lp_full")

    if is_oracle and true_means is None:
        raise ValueError(
            f"acquisition={acquisition!r} needs the TRUE generative means, which "
            f"exist only for the synthetic datasets. Pass "
            f"true_means=core.optimal_static.synthetic_true_means(...), or use "
            f"acquisition='lp_full' for the learned-estimate version of the same "
            f"action space.")

    total_samples = len(x)
    T2 = total_samples - T1
    if T2 <= 0:
        _log.warning("Stage 2 has NO rounds (T1=%d >= n_train=%d) -- returning an "
                     "empty result; the caller will record this cell as a failure",
                     T1, total_samples)
        return {}

    costs = np.asarray(costs, dtype=np.float64)
    nviews = centers.shape[1]
    nclasses = centers.shape[0]

    est_centers = np.array(centers, dtype=np.float64, copy=True)
    est_counts = stage1_counts(y, T1, nclasses, nviews)

    free_indices = [i for i in range(nviews) if costs[i] == 0]
    free_only_mask = np.zeros(nviews, dtype=bool)
    free_only_mask[free_indices] = True
    if not free_only_mask.any():  # no free view in this cost model
        free_only_mask[int(np.argmin(costs))] = True

    stage1_cost_per_sample = float(np.sum(costs))
    stage1_budget_spent = float(T1) * stage1_cost_per_sample
    training_budget_spent = stage1_budget_spent
    training_remaining_budget = float(training_budget - stage1_budget_spent)
    remaining_training_budget_per_sample = training_remaining_budget / T2

    if training_remaining_budget <= 0:
        # Stage 1 consumed the entire training pool, so Stage 2 is the
        # free-view fallback for every one of its T2 rounds regardless of
        # acquisition mode. A legitimate configuration (high init_fraction,
        # low budget_fraction) whose rows are nonetheless not a read on the
        # acquisition policy at all -- worth saying out loud.
        _log.warning("Stage 1 (T1=%d) consumed the whole training budget "
                     "(%.4f spent of %.4f); all %d Stage-2 rounds will use the "
                     "free-view fallback and acquisition=%r is inert",
                     T1, stage1_budget_spent, training_budget, T2, acquisition)

    # ── empirical/enumerated arm state ──
    # Used by lp_chain/lp_full/ucb_argmax/hedge and by greedy when
    # reward_estimate="empirical".
    combos = bit_index = r_hat = combo_counts = None
    combo_masks = combo_cost = cost_order = None

    empirical_est = (reward_estimate == "empirical" and acquisition in ("greedy", "lp_chain"))
    if (reward_estimate == "empirical" and acquisition not in ("greedy", "lp_chain")):
        raise ValueError(
            f"reward_estimate='empirical' applies to acquisition='greedy' "
            f"or 'lp_chain'; got acquisition={acquisition!r}."
        )
    # Arm-elimination compatibility.
    if arm_elimination:
        if acquisition == "lp_full_opt":
            raise ValueError(
                "arm_elimination is not supported with "
                "acquisition='lp_full_opt'. Its fixed oracle distribution "
                "does not use the active-arm mask."
            )
        if acquisition == "greedy" and not empirical_est:
            raise ValueError(
                "Two-Stage arm elimination with acquisition='greedy' "
                "requires reward_estimate='empirical'. Surrogate greedy "
                "does not maintain an enumerated empirical arm table."
            )
    if empirical_est or uses_full_empirical_table:
        # A permanent full arm universe is required by:
        #   1. greedy/lp_chain with reward_estimate="empirical";
        #   2. ucb_argmax and hedge, which always use empirical arm rewards.
        if nviews > MAX_REWARD_ESTIMATE_VIEWS:
            raise ValueError(
                "The selected acquisition/reward-estimate configuration "
                "enumerates 2^(nviews-1) arms; "
                f"nviews={nviews} exceeds "
                f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. "
                "Reduce max_modalities or use a non-enumerated configuration."
            )
        combos = generate_view_combinations(nviews)
        tables = build_arm_tables(combos, costs, nviews,)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        r_hat, combo_counts = stage1_combo_rewards(x, y, est_centers, T1, combos, nviews, pred_rule=pred_rule,)
    elif use_lp:
        # Biased branch reward estimates initialization for each arm from Stage-1 empirical accuracy.
        # This will be used by both lp_chain and lp_full acquisition modes.
        if acquisition == "lp_chain":
            combos = greedy_chain(est_centers, costs, free_indices, force_free=force_free,)
        else:
            combos = generate_view_combinations(nviews)
        tables = build_arm_tables(combos, costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]

        if is_oracle:
            true_means = np.asarray(true_means, dtype=np.float64)
            r_hat = arm_accuracies_from_means(x[T1:], y[T1:], true_means, combo_masks)
            combo_counts = np.full(len(combos), float(T2))
        else:
            r_hat, combo_counts = stage1_combo_rewards(x, y, est_centers, T1, combos, nviews, pred_rule=pred_rule)

    _log.debug("Stage 2 start: acquisition=%s, reward_estimate=%s, T1=%d, T2=%d, "
               "n_arms=%s, budget left %.4f (%.6f/round)",
               acquisition, reward_estimate, T1, T2,
               (len(combos) if combos is not None else 0),
               training_remaining_budget, remaining_training_budget_per_sample)

    lambda_t = 0.0

    predictions = []
    score_rows = []
    true_labels = []
    reward_trace = []
    lagrangian_reward_trace = []
    views_trace = []
    seen_masks = set()
    # Stage 1 observes every modality by definition. Stage 2 entries are
    # appended only after the hard remaining-budget fallback, so this is the
    # actual acquired subset for every training sample, in training order.
    selected_subsets = [tuple(range(1, nviews + 1)) for _ in range(T1)]
    round_trace = [] if trace_rounds else None
    n_budget_fallbacks = 0
    errors = 0

    p_oracle = None
    if is_oracle:
        p_oracle, lambda_t = linprog_policy_over_estimates(
            r_hat, combo_cost, max(0.0, remaining_training_budget_per_sample))
    if is_hedge:
        hedge_v = np.ones(2, dtype=np.float64)
        hedge_epsilon = np.sqrt(np.log(2.0) / T2)

    elimination_trace = []
    if arm_elimination and combo_masks is not None:
        active_arms = np.ones(len(combo_masks), dtype=bool)
        initial_arm_count = int(active_arms.sum())
        elimination_points = set(arm_elimination_checkpoints(T2))
    else:
        active_arms = None
        initial_arm_count = 0
        elimination_points = set()

    for t in range(T1, total_samples):
        round_idx = t - T1
        # ============================================================
        # 1. ACQUISITION
        #    NOT ticked here -- core.submodular_greedy's policy routines
        #    carry @timed("t_acquisition") themselves, and a tick around
        #    this block would count the same span a second time.
        # ============================================================
        if training_remaining_budget <= 0:
            # Budget exhausted: free-view fallback is absorbing.
            mask = free_only_mask.copy()
        #elif (acquisition == "greedy" and training_remaining_budget + 1e-12 >= (total_samples - t) * float(np.sum(costs))):
        #    # Temporary high-budget diagnostic
        #    mask = np.ones(nviews, dtype=bool)
        elif is_oracle:
            # lp_full_opt: fixed oracle distribution solved before Stage 2.
            j = int(rng.choice(len(p_oracle), p=p_oracle))
            mask = combo_masks[j].copy()
        elif is_argmax:
            # These are GLOBAL arm-table indices.
            affordable_idx = np.flatnonzero(combo_cost <= training_remaining_budget + 1e-12)
            candidate_idx = restrict_candidates(affordable_idx, active_arms,)
            # Calculate confidence bounds only for active, affordable arms.
            candidate_cost = combo_cost[candidate_idx]
            candidate_ucb = (r_hat[candidate_idx] + np.sqrt(alpha_ucb * np.log(round_idx + 2)/ combo_counts[candidate_idx]))
            # j_local indexes the sliced candidate arrays.
            j_local = argmax_policy_over_estimates(candidate_ucb, candidate_cost, lambda_t, training_remaining_budget,)
            # Convert to the permanent/global arm-table index.
            j_global = int(candidate_idx[j_local])            
            mask = combo_masks[j_global].copy()
        elif is_hedge:
            # GLOBAL indices of affordable and active arms.
            affordable_idx = np.flatnonzero(combo_cost <= training_remaining_budget + 1e-12)
            candidate_idx = restrict_candidates(affordable_idx, active_arms,)
            candidate_cost = combo_cost[candidate_idx]
            # Calculate UCB only for active candidates.
            candidate_ucb = (r_hat[candidate_idx] + np.sqrt(alpha_ucb * np.log(round_idx + 2)/ combo_counts[candidate_idx]))
            hedge_y = hedge_v / hedge_v.sum()
            candidate_effective_cost = (hedge_y[0] + hedge_y[1] * candidate_cost)
            candidate_score = (candidate_ucb / candidate_effective_cost)
            # Local index into candidate_score.
            j_local = int(np.argmax(candidate_score))
            # Permanent/global arm-table index.
            j_global = int(candidate_idx[j_local])
            # Two-Stage result.
            mask = combo_masks[j_global].copy()
        elif acquisition == "lp_chain" and empirical_est:
            # EMPIRICAL LP-CHAIN
            # Compute this scalar once for the current round.
            log_bonus = (alpha_ucb * np.log(round_idx + 2))
            def gain_func(sel):
                # `sel` contains zero-based view indices.
                bits = sum(1 << int(i) for i in sel)
                j_global = bit_index[bits]
                # Calculate UCB only for the subset requested by greedy_chain.
                return float(r_hat[j_global] + np.sqrt(log_bonus / combo_counts[j_global]))
            chain = greedy_chain(est_centers, costs, free_indices, force_free=force_free, gain_func=gain_func, empty_value=1.0 / nclasses,)
            # Map every chain arm to the permanent/global full-arm table.
            act_idx = np.array([bit_index[sum(1 << (v - 1) for v in combo)] for combo in chain], dtype=int,)
            # Remove eliminated arms from the final LP candidate set.
            candidate_idx = restrict_candidates(act_idx, active_arms,)
            # Calculate UCB only for active arms that survived chain filtering.
            candidate_ucb = (r_hat[candidate_idx] + np.sqrt(log_bonus / combo_counts[candidate_idx]))
            candidate_cost = combo_cost[candidate_idx]
            b_t = (max(0.0, training_remaining_budget) / max(1, total_samples - t))
            p_lp, lambda_t = linprog_policy_over_estimates(candidate_ucb, candidate_cost, b_t,)
            # Local LP index.
            j_local = int(rng.choice(len(p_lp), p=p_lp,))
            # Permanent/global arm-table index.
            j_global = int(candidate_idx[j_local])
            mask = combo_masks[j_global].copy()
        elif use_lp:
            # SURROGATE LP-CHAIN / LP-FULL
            # GLOBAL indices into combo_masks/r_hat/combo_counts.
            candidate_idx = restrict_candidates(np.arange(len(combo_masks), dtype=int), active_arms,)
            # Calculate confidence bounds only for the active candidates.
            candidate_ucb = (r_hat[candidate_idx] + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts[candidate_idx]))
            candidate_cost = combo_cost[candidate_idx]
            b_t = (max(0.0, training_remaining_budget) / max(1, total_samples - t))
            p_lp, lambda_t = linprog_policy_over_estimates(candidate_ucb, candidate_cost, b_t,)
            # Local index into the sliced arrays.
            j_local = int(rng.choice(len(p_lp), p=p_lp))
            # Map back to the permanent/global arm table.
            j_global = int(candidate_idx[j_local])
            mask = combo_masks[j_global].copy()
        elif empirical_est:
            # EMPIRICAL GREEDY
            log_bonus = (alpha_ucb * np.log(round_idx + 2))
            # Preserve the original unconstrained behavior until at least
            # one arm has actually been eliminated.
            if (active_arms is not None and not np.all(active_arms)):
                greedy_active_bits = arm_bits[active_arms]
            else:
                greedy_active_bits = None
            def gain_func(sel):
                bits = sum(1 << int(i) for i in sel)
                j_global = bit_index[bits]
                return float(r_hat[j_global] + np.sqrt(log_bonus / combo_counts[j_global]))
            mask = greedy_oracle(
                None,
                costs,
                lambda_t,
                training_remaining_budget,
                free_indices,
                force_free=force_free,
                gain_func=gain_func,
                empty_value=1.0 / nclasses,
                active_arm_bits=greedy_active_bits,
            )
        else:
            # SURROGATE GREEDY
            diff_mean_sq = pairwise_diff_sq_from_means(est_centers)
            if alpha_ucb > 0:
                inv_sqrt_cnt = np.sqrt(1.0 / est_counts).T
                bonus = (inv_sqrt_cnt[:, :, None]+ inv_sqrt_cnt[:, None, :])
                bonus *= np.sqrt(alpha_ucb * np.log(round_idx + 2))
                diff_mean_sq = diff_mean_sq + bonus
            mask = greedy_oracle(diff_mean_sq, costs, lambda_t, training_remaining_budget, free_indices, force_free=force_free,)

        # ============================================================
        # 2. HARD BUDGET CHECK
        # ============================================================
        cost = float(np.sum(costs[mask]))
        if training_remaining_budget < cost:
            mask = free_only_mask.copy()
            cost = float(np.sum(costs[mask]))
            n_budget_fallbacks += 1
        if is_hedge:
            resource_z = np.array([1.0, cost], dtype=np.float64,)
            hedge_v *= ((1.0 + hedge_epsilon) ** resource_z)
        # ============================================================
        # 3. PREDICTION / REWARD
        # ============================================================
        with tick("t_predict"):
            y_hat, score = predict_mask_multiclass(x[t], est_centers, mask, return_score=True, pred_rule=pred_rule)
        y_true = int(y[t])
        reward = float(y_hat == y_true)
        lagrangian_reward = reward - lambda_t * cost

        predictions.append(y_hat)
        score_rows.append(score)
        true_labels.append(y_true)
        reward_trace.append(reward)
        lagrangian_reward_trace.append(lagrangian_reward)
        views_trace.append(int(mask.sum()))
        seen_masks.add(mask.tobytes())
        selected_subsets.append(tuple((np.flatnonzero(mask) + 1).tolist()))
        errors += int(y_hat != y_true)

        # ============================================================
        # 4. EMPIRICAL ARM-REWARD UPDATE
        # ============================================================

        if (use_lp or empirical_est or is_argmax or is_hedge) and not is_oracle:
            with tick("t_reward_update"):
                played_bits = mask_to_bits(mask)
                if reward_update == "selected":
                    # Only the actually played subset receives an observation.
                    # j0 = bit_index.get(played_bits)
                    # targets = ([] if j0 is None else [(int(j0), reward)])
                    j0 = bit_index.get(played_bits)
                    if (j0 is not None and (active_arms is None or active_arms[j0])):
                        targets = [(j0, float(reward))]
                    else:
                        targets = []
                else:
                    # Every arm contained in the acquired subset is observable.
                    targets = []
                    candidate_idx = restrict_candidates(np.arange(len(arm_bits), dtype=int), active_arms,)
                    candidate_bits = arm_bits[candidate_idx]
                    contained_local = ((candidate_bits & played_bits) == candidate_bits)
                    contained_idx = candidate_idx[contained_local]
                    for k in contained_idx:
                        m_k = combo_masks[k]
                        y_sub = predict_mask_multiclass(x[t], est_centers, m_k, pred_rule=pred_rule,)
                        targets.append((int(k), float(y_sub == y_true),))
                for j0, r_obs in targets:
                    combo_counts[j0] += 1.0
                    r_hat[j0] += (r_obs - r_hat[j0] ) / combo_counts[j0]

        # ============================================================
        # 5. DUAL UPDATE
        # ============================================================
        if acquisition in ("greedy",) + ARGMAX_ACQUISITION_MODES:
            with tick("t_dual_update"):
                raw_lambda = (lambda_t + step_size * (cost - remaining_training_budget_per_sample))
                lambda_t = max(0.0, min(lambda_max, raw_lambda),)
        # ============================================================
        # 6. BUDGET BOOKKEEPING
        # ============================================================
        training_remaining_budget -= cost
        training_budget_spent += cost

        if round_trace is not None:
            # Recorded AFTER the bookkeeping so remaining_budget is the
            # post-round figure -- the one that explains the NEXT round's
            # choice, which is what makes a trace readable top to bottom.
            round_trace.append({
                "round": round_idx,
                "t": t,
                "subset": (np.flatnonzero(mask) + 1).tolist(),
                "n_views": int(mask.sum()),
                "cost": cost,
                "lambda": float(lambda_t),
                "reward": reward,
                "lagrangian_reward": lagrangian_reward,
                "remaining_budget": float(training_remaining_budget),
            })

        completed_stage2 = t - T1 + 1
        if (arm_elimination and active_arms is not None and completed_stage2 in elimination_points):
            remaining_rounds = T2 - completed_stage2
            b_elim = (max(0.0, training_remaining_budget) / max(1, remaining_rounds))
            active_arms, elim_info = eliminate_arms_ucb_lcb(r_hat=r_hat, combo_counts=combo_counts, combo_cost=combo_cost, active_arms=active_arms, alpha_ucb=alpha_ucb, round_idx=completed_stage2, budget_per_round=b_elim,)
            _log.info("[arm elimination] Stage2=%d/%d: %d -> %d active arms (removed %d)", completed_stage2, T2, elim_info["before"], elim_info["after"], elim_info["eliminated"],)
            elimination_trace.append({
                "completed_stage2_rounds": int(completed_stage2),
                "before": int(elim_info["before"]),
                "after": int(elim_info["after"]),
                "eliminated": int(elim_info["eliminated"]),
                "budget_per_remaining_round": float(b_elim),
            })


    if n_budget_fallbacks:
        _log.info("Stage 2: %d/%d rounds (%.1f%%) hit the hard budget check and fell "
                  "back to the free view", n_budget_fallbacks, T2,
                  100.0 * n_budget_fallbacks / max(1, T2))

    error_rate = errors / len(reward_trace)
    avg_reward = float(np.mean(reward_trace))
    avg_lagrangian_reward = float(np.mean(lagrangian_reward_trace))

    score_mat = np.asarray(score_rows)
    train_f1 = _macro_f1(true_labels, predictions)
    train_auroc = _macro_ovr_auroc(np.asarray(true_labels), score_mat, nclasses)

    return {
        'T2': T2,
        'error_rate': error_rate,
        'avg_reward': avg_reward,
        'avg_lagrangian_reward': avg_lagrangian_reward,
        'train_f1': train_f1,
        'train_auroc': train_auroc,
        'training_budget_spent': training_budget_spent,
        'training_budget_spent_stage2': training_budget_spent - stage1_budget_spent,
        'training_remaining_budget': training_remaining_budget,
        'lambda_final': lambda_t,
        'warm_start': False,
        'centers': est_centers,
        'combo_rewards': r_hat,
        'combo_counts': combo_counts,
        'est_counts': est_counts,
        'acquisition': acquisition,
        'n_arms': (len(combos) if combos is not None else 0),
        'oracle_probs': p_oracle,
        'avg_views_acquired': float(np.mean(views_trace)),
        'n_unique_masks': len(seen_masks),
        'selected_subsets': selected_subsets,
        'round_trace': round_trace,
        'n_budget_fallbacks': n_budget_fallbacks,
        # Elimination
        'arm_elimination': bool(arm_elimination),
        'initial_arms': int(initial_arm_count),
        'final_active_arms': (int(active_arms.sum()) if active_arms is not None else int(initial_arm_count)),
        'num_eliminated': (int(initial_arm_count - active_arms.sum()) if active_arms is not None else 0),
        'elimination_trace': elimination_trace,
    }