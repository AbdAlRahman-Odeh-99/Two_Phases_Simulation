# -*- coding: utf-8 -*-
"""
two_stage_multiclass_greedy.py

=== acquisition: which subset the policy buys each round ===
two_stage's premise is that the CLASSIFIER is frozen after Stage 1 --
the centres learned there are what Stage 2 predicts with and what
inference predicts with, unchanged. This module keeps that premise under
EVERY mode, so the only axis left is the ACQUISITION policy, and it is
spelled exactly as gmm_multiclass_submodular spells it (both read
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

=== UCB CAPPING ===
Every r_hat in this module is an EMPIRICAL 0/1 ACCURACY, so it lives in
[0, 1] and so does its optimistic counterpart. The exploration bonus
sqrt(alpha_ucb * log(round + 2) / count) does NOT respect that range: at
count = T1 = 0 (init_fraction 0, where stage1_combo_rewards can only
return a flat 1/nclasses with count 1) the bonus passes 1.0 within a few
dozen rounds and exceeds the entire range of the quantity it decorates.
Arm ORDERING is then driven by which arms the containment replay has
happened to touch, not by learned accuracy, and the policy degenerates
into "buy the largest affordable subset". Every UCB below is therefore
capped at 1.0, which is the correct ceiling for a probability and keeps
the ordering meaningful once counts diverge. The cap applies to the
BIASED lp_chain/lp_full paths too -- their r_hat is the same empirical
accuracy table, seeded by stage1_combo_rewards.
"""

from __future__ import annotations

import numpy as np

from core.two_stage_utils import generate_view_combinations
from core.submodular_greedy import (
    # Shared policy vocabulary -- core is the canonical definition, so this
    # method and gmm_multiclass_submodular agree on what run_proposed_
    # methods.py's --acquisition / --reward-update values mean.
    ACQUISITION_MODES,           # noqa: F401 -- re-exported
    LP_ACQUISITION_MODES,        # noqa: F401 -- re-exported
    MAX_REWARD_ESTIMATE_VIEWS,   # noqa: F401 -- re-exported
    ORACLE_ACQUISITION_MODES,    # noqa: F401 -- re-exported
    REWARD_UPDATE_SCOPES,        # noqa: F401 -- re-exported
    pairwise_diff_sq_from_means,
    arm_accuracies_from_means,
    greedy_chain,                # noqa: F401 -- re-exported (moved here)
    greedy_oracle,
    lp_policy_over_estimates,    # noqa: F401 -- re-exported (moved here)
    linprog_policy_over_estimates,
    build_arm_tables,
    mask_to_bits,
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

REWARD_ESTIMATES = ("biased", "unbiased")

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

def stage1_combo_rewards(x, y, centers, T1, combos, nviews):
    """
    Stage 1 observes ALL views for its T1 rows, so the frozen classifier can
    be REPLAYED over those rows using only the views in a combination. The
    result is that combination's EMPIRICAL ACCURACY.
    """
    if T1 <= 0:
        return (np.full(len(combos), 1.0 / centers.shape[0]), np.ones(len(combos), dtype=np.float64))
    xs, ys = x[:T1], np.asarray(y[:T1], dtype=int)
    r_hat = np.empty(len(combos), dtype=np.float64)
    for j, combo in enumerate(combos):
        m = np.zeros(nviews, dtype=bool)
        m[np.array(combo) - 1] = True
        d = ((xs[:, None, m] - centers[None, :, m]) ** 2).sum(axis=2)  # (T1, nc)
        r_hat[j] = float(np.mean(d.argmin(axis=1) == ys))
    return r_hat, np.full(len(combos), float(T1))


# Stage 2 (training phase)
def run_alg_greedy_multiclass(x, y, centers, costs, T1, training_budget, rng,
                              acquisition="greedy", alpha_ucb=2.0,
                              step_size=1.0, lambda_max=10.0,
                              pred_rule="nearest_center",
                              force_free=True, reward_update="subsets",
                              true_means=None, reward_estimate="biased"):
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
    acquisition : {"greedy", "lp_chain", "lp_full", "lp_full_opt"}
        The per-round acquisition policy -- see the module docstring.
        Default "greedy". Same vocabulary as
        gmm_multiclass_submodular.run_training_phase's flag of the same
        name (both validate against core.submodular_greedy's
        ACQUISITION_MODES), which is what makes the two methods comparable
        on this axis.
    reward_update : {"subsets", "selected"}
        Arm-scoring scope, lp_chain/lp_full ONLY; inert under "greedy",
        which has no enumerated arms, and under "lp_full_opt", whose arm
        values are exact and never scored.
    reward_estimate : {"biased", "unbiased"}
        How greedy_oracle / greedy_chain value a subset. Validated for
        MEMBERSHIP here, not only for compatibility with `acquisition`: a
        misspelling used to fall through the `reward_estimate ==
        "unbiased"` test and silently run the BIASED path under an
        unbiased label.
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
        gmm_multiclass_submodular.run_training_phase. alpha_ucb=0 makes
        the oracle purely exploitative (and, under acquisition="greedy",
        fully deterministic given lambda). The resulting UCB is capped at
        1.0 -- see the module docstring's UCB CAPPING section.
    force_free : bool, default True
        Passed through to greedy_oracle -- keeps the free view(s) in every
        acquired subset, matching EXP4's and the LP's invariant.

    Returns
    -------
    dict with the SAME keys run_alg_multiclass returns (so the runner and
    run_proposed_methods.normalize_two_stage_mc need no changes), plus
    'centers', 'est_counts', 'acquisition', 'avg_views_acquired',
    'n_unique_masks'. 'warm_start' is always False -- there is no expert
    weight vector to warm-start here; the Stage-1 information enters
    through the centres and (via stage1_counts) the optimism bonus instead.
    """
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                         f"got {acquisition!r}")
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")
    # MEMBERSHIP check, deliberately separate from the acquisition-
    # compatibility check further down. Without it a typo ("unbaised")
    # makes the `reward_estimate == "unbiased"` tests below evaluate False
    # and the run proceeds on the BIASED path -- reported, and filenamed,
    # as whatever string was passed. Silent wrong-mode runs are worse than
    # a crash, so this rejects rather than falls back.
    if reward_estimate not in REWARD_ESTIMATES:
        raise ValueError(f"reward_estimate must be one of {REWARD_ESTIMATES}, "
                         f"got {reward_estimate!r}")
    # One boolean, computed once, replacing the old
    # `center_update == "reward_estimates"` test at each of its five sites.
    use_lp = acquisition in LP_ACTION_SPACE
    # Second boolean: an oracle mode enumerates and plays like an LP mode but
    # neither estimates nor re-solves, so it splits off from `use_lp` at
    # exactly two sites -- the acquisition branch and the r_hat update.
    is_oracle = acquisition in ORACLE_ACQUISITION_MODES
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

    training_budget_spent = float(T1)
    training_remaining_budget = float(training_budget - T1)
    remaining_training_budget_per_sample = training_remaining_budget / T2
    # ── empirical/enumerated arm state ──
    # Used by lp_chain/lp_full and by greedy when reward_estimate="unbiased".
    combos = bit_index = r_hat = combo_counts = None
    combo_masks = combo_cost = cost_order = None

    unbiased = (reward_estimate == "unbiased" and acquisition in ("greedy", "lp_chain"))
    if (reward_estimate == "unbiased" and acquisition not in ("greedy", "lp_chain")):
        raise ValueError(
            f"reward_estimate='unbiased' applies to acquisition='greedy' "
            f"or 'lp_chain'; got acquisition={acquisition!r}."
        )

    if unbiased:
        # Unbiased branch reward estimates intialization for each arm from Stage-1 empirical accuracy.
        # This will be used by both greedy and lp_chain acquisition modes.
        if nviews > MAX_REWARD_ESTIMATE_VIEWS:
            raise ValueError(
                f"reward_estimate='unbiased' enumerates "
                f"2^(nviews-1) = 2^{nviews - 1} arms; "
                f"nviews={nviews} exceeds "
                f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}."
            )
        # Full arm universe, ONCE.
        combos = generate_view_combinations(nviews)
        tables = build_arm_tables(combos, costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        # Stage 1 is fully observed, so initialize EVERY arm from genuine Stage-1 empirical accuracy
        r_hat, combo_counts = stage1_combo_rewards(x, y, est_centers, T1, combos, nviews)
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
            r_hat, combo_counts = stage1_combo_rewards(x, y, est_centers, T1, combos, nviews)

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
    errors = 0

    # ── ORACLE LP, solved ONCE before the loop ("lp_full_opt" only) ──
    # Exact arm values and a flat allowance mean nothing about this LP
    # changes between rounds, so it is solved here and every Stage-2 round
    # reduces to one draw from p_oracle. The allowance is the FLAT
    # (training_budget - T1) / T2 and stays flat: "solve once" and "re-solve
    # against what is left" cannot both hold, and this mode is the former by
    # construction. That makes it the ceiling for NON-ADAPTIVE policies --
    # it never looks at x_t or at surviving budget -- so the adaptive modes
    # below may legitimately beat it. Same caveat core.optimal_static states.
    p_oracle = None
    if is_oracle:
        p_oracle, lambda_t = linprog_policy_over_estimates(
            r_hat, combo_cost, max(0.0, remaining_training_budget_per_sample))

    for t in range(T1, total_samples):
        round_idx = t - T1
        # ============================================================
        # 1. ACQUISITION
        # ============================================================
        if training_remaining_budget <= 0:
            # Budget exhausted: free-view fallback is absorbing.
            mask = free_only_mask.copy()
        elif is_oracle:
            # lp_full_opt: fixed oracle distribution solved before Stage 2.
            j = int(rng.choice(len(p_oracle), p=p_oracle))
            mask = combo_masks[j].copy()
        elif acquisition == "lp_chain" and unbiased:
            # --------------------------------------------------------
            # UNBIASED LP-CHAIN
            #
            # r_hat/combo_counts live on the FULL powerset table.
            # Build the current greedy chain using those empirical UCB
            # values, then solve the LP only over the chain.
            # --------------------------------------------------------
            # capped at 1.0 -- r_hat is an accuracy (module docstring).
            ucb_full = r_hat + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts)
            def gain_func(sel, _u=ucb_full):
                bits = sum(1 << int(i) for i in sel)
                return float(_u[bit_index[bits]])
            chain = greedy_chain(est_centers, costs, free_indices, force_free=force_free, gain_func=gain_func, empty_value=1.0 / nclasses,)
            # Map the current chain back into the permanent full table.
            act_idx = np.array([bit_index[sum(1 << (v - 1) for v in combo)] for combo in chain], dtype=int,)
            act_masks = combo_masks[act_idx]
            act_cost = combo_cost[act_idx]
            act_ucb = ucb_full[act_idx]
            b_t = (max(0.0, training_remaining_budget) / max(1, total_samples - t))
            p_lp, lambda_t = linprog_policy_over_estimates(act_ucb, act_cost, b_t,)
            j = int(rng.choice(len(p_lp), p=p_lp))
            mask = act_masks[j].copy()
        elif use_lp:
            # --------------------------------------------------------
            # BIASED LP-CHAIN / LP-FULL
            #
            # For biased lp_chain the arm table is the fixed
            # Bhattacharyya-derived chain constructed before Stage 2.
            # For lp_full it is the full enumeration.
            # --------------------------------------------------------
            ucb = r_hat + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts)
            b_t = (max(0.0, training_remaining_budget) / max(1, total_samples - t))
            p_lp, lambda_t = linprog_policy_over_estimates(ucb, combo_cost, b_t,)
            j = int(rng.choice(len(p_lp), p=p_lp))
            mask = combo_masks[j].copy()
        elif unbiased:
            # --------------------------------------------------------
            # UNBIASED GREEDY
            #
            # Search with empirical subset UCB rather than the
            # Bhattacharyya surrogate.
            # --------------------------------------------------------
            # capped at 1.0 -- r_hat is an accuracy (module docstring).
            ucb = r_hat + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts)
            def gain_func(sel, _ucb=ucb):
                bits = sum(1 << int(i) for i in sel)
                return float(_ucb[bit_index[bits]])
            mask = greedy_oracle(None, costs, lambda_t, training_remaining_budget, free_indices, force_free=force_free, gain_func=gain_func, empty_value=1.0 / nclasses,)
        else:
            # --------------------------------------------------------
            # BIASED GREEDY
            #
            # NOTE: this bonus decorates diff_mean_sq (a squared distance),
            # NOT a probability, so it is deliberately NOT capped at 1.0.
            # --------------------------------------------------------
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
        # ============================================================
        # 3. PREDICTION / REWARD
        # ============================================================
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

        if (use_lp or unbiased) and not is_oracle:
            played_bits = mask_to_bits(mask)
            if reward_update == "selected":
                # Only the actually played subset receives an observation.
                j0 = bit_index.get(played_bits)
                targets = ([] if j0 is None else [(int(j0), reward)])
            else:
                # Every arm contained in the acquired subset is observable.
                contained_idx = np.flatnonzero((arm_bits & played_bits) == arm_bits)
                targets = []
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
        if not use_lp:
            raw_lambda = (lambda_t + step_size * (cost - remaining_training_budget_per_sample))
            lambda_t = max(0.0, min(lambda_max, raw_lambda),)
        # ============================================================
        # 6. BUDGET BOOKKEEPING
        # ============================================================
        training_remaining_budget -= cost
        training_budget_spent += cost


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
        'training_budget_spent_stage2': training_budget_spent - T1,
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
    }