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

The lp_* modes learn per-ARM reward estimates (r_hat), not centres, so
nothing about the classifier moves under any of the three.

=== HISTORY: the deleted centre-learning modes ===
This file used to carry a second axis, `center_update`, with four modes
that let Stage 2 keep learning centres -- "reward_full" / "reward_bandit"
(a SEPARATE acq_centers estimate feeding the oracle while the classifier
stayed frozen) and "full" / "bandit" (gmm_multiclass_submodular's
feedback rules applied verbatim, classifier included). They were never
used in any reported result and have been REMOVED, along with the
acq_centers array, the `lr` complementary-label step, and the
resolve_acquisition/acquisition_of translation layer that mapped
`acquisition` onto the old (center_update, action_space) pair. The three
surviving policies are named directly. Old mode -> new name, for reading
pre-refactor result files:

    center_update="frozen"                             -> "greedy"
    center_update="reward_estimates", action_space=... -> "lp_chain"/"lp_full"
    center_update="reward_full"/"reward_bandit"/"full"/"bandit"
                                                       -> no equivalent

NOTE that "reward_full" was this function's historical DEFAULT, so an old
unflagged call is NOT reproduced by the new unflagged (acquisition=
"greedy") call.

=== Indexing ===
two_stage_multiclass.py speaks in 1-INDEXED combo TUPLES (view 1 forced in
every combo); the greedy oracle and the LP solver speak in 0-INDEXED
BOOLEAN MASKS over a per-view `costs` array. This file works natively in
mask space and only exposes predict_mask_multiclass; the combo-tuple
plumbing (generate_view_combinations /
generate_combination_costs_heterogeneous) is not imported and not needed.
"""

from __future__ import annotations

import numpy as np

from core.lp_colgen import pairwise_diff_sq_from_means
from core.two_stage_utils import generate_view_combinations
from core.submodular_greedy import (
    # Shared policy vocabulary -- core is the canonical definition, so this
    # method and gmm_multiclass_submodular agree on what run_proposed_
    # methods.py's --acquisition / --reward-update values mean.
    ACQUISITION_MODES,           # noqa: F401 -- re-exported
    MAX_REWARD_ESTIMATE_VIEWS,   # noqa: F401 -- re-exported
    REWARD_UPDATE_SCOPES,        # noqa: F401 -- re-exported
    greedy_chain,                # noqa: F401 -- re-exported (moved here)
    greedy_oracle,
    lp_policy_over_estimates,    # noqa: F401 -- re-exported (moved here)
    multiclass_risk,
)

# These six lived in two_stage/two_stage_multiclass.py until that module
# was deleted along with the EXP4 `two_stage` method; they were never
# EXP4-specific, so they moved VERBATIM to core/multiclass_common.py.
from core.multiclass_common import (
    PRED_RULES,                                  # noqa: F401 -- re-exported
    _class_posterior_scores,
    _macro_f1,
    _macro_ovr_auroc,
    _pred_nearest_center,
    _pred_pairwise_vote,
)

# The action space of the two LP modes is carried by the mode NAME
# ("lp_chain" / "lp_full"), so there is no separate action_space flag:
# LP_ACTION_SPACE[acquisition] is the single place that mapping lives.
LP_ACTION_SPACE = {"lp_chain": "chain", "lp_full": "full"}


# ─────────────────────────────────────────────────────────────────────────
# Mask-space prediction -- same two rules, same shared posterior scores as
# two_stage_multiclass.predict_single_combination_multiclass, just taking a
# 0-indexed boolean mask instead of a 1-indexed combo tuple.
# ─────────────────────────────────────────────────────────────────────────
def predict_mask_multiclass(x_sample, centers, mask, return_score=False,
                            pred_rule="nearest_center"):
    """K-class prediction over the views selected by `mask` (0-indexed bool,
    length nviews). See two_stage_multiclass.py's PRED_RULES section for why
    the two rules are mathematically equivalent."""
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
    """Reconstruct the per-(class, view) visitation counts implied by
    Stage 1, WITHOUT modifying initialize_centers_multiclass.

    Stage 1 observes ALL views for each of its T1 samples and updates the
    true class's centre, starting from counts == 1 (the rng.normal draw
    counted as one pseudo-observation -- see initialize_centers_multiclass,
    which blends via `1 - 1/(counts+1)`). So class k's count after Stage 1
    is 1 + #{i < T1 : y_i == k}, identically for every view.
    """
    per_class = 1 + np.bincount(np.asarray(y[:T1], dtype=int), minlength=nclasses)
    return np.repeat(per_class[:nclasses, None], nviews, axis=1).astype(np.float64)


def stage1_combo_rewards(x, y, centers, T1, combos, nviews):
    """EAGER per-combination reward estimate after Stage 1, plus its counts.

    Stage 1 observes ALL views for its T1 rows, so the frozen classifier can
    be REPLAYED over those rows using only the views in a combination. The
    result is that combination's EMPIRICAL ACCURACY -- genuine observations
    on exactly the scale the Stage-2 updates use, which is why the counts
    start at T1 ("counts continue from Stage 1"), not at a pseudo-count.

    Vectorised nearest-centre evaluation, valid for BOTH pred_rules:
    nearest_center and pairwise_vote were verified to agree on 3840/3840
    (combo, sample) pairs, as two_stage_multiclass.py's PRED_RULES section
    states.
    """
    if T1 <= 0:
        # Nothing to replay: uninformative prior at chance level.
        return (np.full(len(combos), 1.0 / centers.shape[0]),
                np.ones(len(combos), dtype=np.float64))
    xs, ys = x[:T1], np.asarray(y[:T1], dtype=int)
    r_hat = np.empty(len(combos), dtype=np.float64)
    for j, combo in enumerate(combos):
        m = np.zeros(nviews, dtype=bool)
        m[np.array(combo) - 1] = True
        d = ((xs[:, None, m] - centers[None, :, m]) ** 2).sum(axis=2)  # (T1, nc)
        r_hat[j] = float(np.mean(d.argmin(axis=1) == ys))
    return r_hat, np.full(len(combos), float(T1))

# ─────────────────────────────────────────────────────────────────────────
# Stage 2 (training phase): per-round submodular greedy + OMD dual.
#
# The dual update, the budget bookkeeping, the reward definition, the
# Lagrangian trace and the returned dict are BYTE-FOR-BYTE the same logic
# as two_stage_multiclass.run_alg_multiclass -- only the block that chooses
# `mask` (and the optional centre/count update) differs.
# ─────────────────────────────────────────────────────────────────────────
def run_alg_greedy_multiclass(x, y, centers, costs, T1, training_budget, rng,
                              acquisition="greedy", alpha_ucb=2.0,
                              step_size=1.0, lambda_max=10.0,
                              pred_rule="nearest_center",
                              force_free=True, reward_update="subsets"):
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
    acquisition : {"greedy", "lp_chain", "lp_full"}
        The per-round acquisition policy -- see the module docstring.
        Default "greedy". Same vocabulary as
        gmm_multiclass_submodular.run_training_phase's flag of the same
        name (both validate against core.submodular_greedy's
        ACQUISITION_MODES), which is what makes the two methods comparable
        on this axis.
    reward_update : {"subsets", "selected"}
        Arm-scoring scope, lp_chain/lp_full ONLY; inert under "greedy",
        which has no enumerated arms.
    alpha_ucb : float, default 2.0
        Optimism scale in the bonus sqrt(alpha_ucb * log(t+1)) / sqrt(count)
        -- same knob and same default as
        gmm_multiclass_submodular.run_training_phase. alpha_ucb=0 makes
        the oracle purely exploitative (and, under acquisition="greedy",
        fully deterministic given lambda).
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
    # One boolean, computed once, replacing the old
    # `center_update == "reward_estimates"` test at each of its five sites.
    use_lp = acquisition in LP_ACTION_SPACE

    total_samples = len(x)
    T2 = total_samples - T1
    if T2 <= 0:
        return {}

    costs = np.asarray(costs, dtype=np.float64)
    nviews = centers.shape[1]
    nclasses = centers.shape[0]

    # Never mutate the caller's array: the runner reuses `centers` for
    # two_stage_error's init_error and for inference. Nothing here writes
    # to est_centers -- the copy is kept so the returned 'centers' is the
    # caller's to own, and so a future learning mode has one array to touch.
    est_centers = np.array(centers, dtype=np.float64, copy=True)
    # Frozen at their Stage-1 values: no mode increments these any more, so
    # the greedy oracle's optimism bonus grows only through the log(t+1)
    # numerator. This is exactly what the old center_update="frozen" did.
    est_counts = stage1_counts(y, T1, nclasses, nviews)

    free_indices = [i for i in range(nviews) if costs[i] == 0]
    free_only_mask = np.zeros(nviews, dtype=bool)
    free_only_mask[free_indices] = True
    if not free_only_mask.any():  # no free view in this cost model
        free_only_mask[int(np.argmin(costs))] = True

    # SAME budget convention as run_alg_multiclass: Stage 1 is charged 1.0
    # per fully-observed round, and Stage 2's per-round OMD target is what
    # is LEFT divided by T2 (NOT submodular's training_budget/n_train).
    training_budget_spent = float(T1)
    training_remaining_budget = float(training_budget - T1)
    remaining_training_budget_per_sample = training_remaining_budget / T2
    # ── arm setup, lp_chain / lp_full only ──
    combos = bit_index = r_hat = combo_counts = None
    combo_masks = combo_cost = cost_order = None
    start_bits = 0
    if use_lp:
        action_space = LP_ACTION_SPACE[acquisition]
        if action_space == "full":
            # RETIRED as the default; kept for the fidelity check that
            # measures chain-restricted vs. full-enumeration accuracy on
            # identical seeds at small nviews.
            if nviews > MAX_REWARD_ESTIMATE_VIEWS:
                raise ValueError(
                    f"acquisition='lp_full' enumerates 2^(nviews-1) = "
                    f"2^{nviews - 1} combinations eagerly; nviews={nviews} "
                    f"exceeds MAX_REWARD_ESTIMATE_VIEWS="
                    f"{MAX_REWARD_ESTIMATE_VIEWS}. Use --acquisition lp_chain "
                    f"(the default LP mode) or --acquisition greedy, neither "
                    f"of which enumerates, or --max-modalities to trim.")
            combos = generate_view_combinations(nviews)
        else:
            combos = greedy_chain(est_centers, costs, free_indices,
                                  force_free=force_free)
        bit_index = {}
        combo_masks = np.zeros((len(combos), nviews), dtype=bool)
        combo_cost = np.zeros(len(combos), dtype=np.float64)
        for j, combo in enumerate(combos):
            bit_index[sum(1 << (v - 1) for v in combo)] = j
            combo_masks[j, np.array(combo) - 1] = True
            combo_cost[j] = float(costs[combo_masks[j]].sum())
        # costs never change, so the LP's cost ordering is computed ONCE
        cost_order = np.argsort(combo_cost, kind="stable")
        r_hat, combo_counts = stage1_combo_rewards(x, y, est_centers, T1,
                                                   combos, nviews)
        # Start every build from the free views, matching force_free.
        start_bits = sum(1 << i for i in free_indices) or 1
        if start_bits not in bit_index:
            # Under "chain" the smallest arm IS the free set (or the cheapest
            # view when nothing is free), so seed from it. Under "full",
            # generate_view_combinations only forces view 1; if some other
            # view is also free, fall back to view 1 alone.
            start_bits = (min(bit_index, key=lambda b: bin(b).count("1"))
                          if action_space == "chain" else 1)
        # REPLAY TABLE: for each arm, the arms whose view set it contains, so
        # a round that played arm j scores them all for free. Under the chain
        # the arms are nested, so this is O(nviews) per round instead of the
        # 2^|S| submask walk _submasks_containing() performs over the full
        # action space -- and it is built without any enumeration.
        arm_bits = np.array(sorted(bit_index, key=lambda b: bit_index[b]))
        sub_arms = [[int(k) for k in range(len(combos))
                     if (arm_bits[k] & int(b)) == arm_bits[k]]
                    for b in arm_bits]
    lambda_t = 0.0

    predictions = []
    score_rows = []
    true_labels = []
    reward_trace = []
    lagrangian_reward_trace = []
    views_trace = []
    seen_masks = set()
    errors = 0

    b_fixed =  training_remaining_budget / T2
    for t in range(T1, total_samples):
        round_idx = t - T1  # 0-based Stage-2 round, drives the log(t+1) bonus

        if training_remaining_budget <= 0:
            # SAFE BUDGET CHECK (BwK-style absorbing fallback policy) --
            # same semantics as run_alg_multiclass's cheapest-combo branch.
            mask = free_only_mask.copy()
        elif use_lp:
            ucb = r_hat + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts)
            # b_t = max(0.0, training_remaining_budget / max(1, total_samples - t))
            b_t = b_fixed
            i_lo, i_hi, p_hi, lambda_t = lp_policy_over_estimates(
                ucb, combo_cost, cost_order, b_t)
            j = i_hi if (p_hi > 0.0 and float(rng.random()) < p_hi) else i_lo
            mask = combo_masks[j].copy()
        else:
            diff_mean_sq = pairwise_diff_sq_from_means(est_centers)  # (nv, nc, nc)
            if alpha_ucb > 0:
                inv_sqrt_cnt = np.sqrt(1.0 / est_counts).T           # (nv, nc)
                bonus = inv_sqrt_cnt[:, :, None] + inv_sqrt_cnt[:, None, :]
                bonus *= np.sqrt(alpha_ucb * np.log(round_idx + 2))
                diff_mean_sq = diff_mean_sq + bonus
            mask = greedy_oracle(diff_mean_sq, costs, lambda_t,
                                 training_remaining_budget, free_indices,
                                 force_free=force_free)

        cost = float(np.sum(costs[mask]))
        if training_remaining_budget < cost:
            # Free-views-only override (notebook convention, and the same
            # "fall back to the cheapest thing you can still afford" rule
            # the EXP4 version applies).
            mask = free_only_mask.copy()
            cost = float(np.sum(costs[mask]))

        y_hat, score = predict_mask_multiclass(
            x[t], est_centers, mask, return_score=True, pred_rule=pred_rule)

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
        errors += int(y_hat != y_true)

        # ── ARM-REWARD update (lp_chain / lp_full only) ──
        # No centre update of any kind: est_centers and est_counts are the
        # Stage-1 arrays throughout, under every acquisition mode. Under
        # "greedy" there is nothing to update at all -- the oracle is
        # re-derived each round from those frozen arrays and the dual.
        if use_lp:
            sel_bits = int(sum(1 << i for i in range(nviews) if mask[i]))
            if reward_update == "selected":
                # BANDIT feedback: one observation per round, the played arm
                # only, scored by the 0/1 reward already computed above.
                targets = [(sel_bits, reward)]
            else:
                # COUNTERFACTUAL REPLAY. x[t] was paid for on every view in
                # the selected set, so any SUB-combination of it can be
                # scored for free -- 2^(|S|-1) observations per round rather
                # than 1. This needs y_true (a wrong prediction does not
                # reveal it), so this scope is FULL feedback, unlike
                # "selected". It is what makes 2^(nviews-1) arms learnable
                # in n_train rounds at all.
                targets = []
                for k in sub_arms[bit_index[sel_bits]]:
                    y_sub = predict_mask_multiclass(x[t], est_centers,
                                                    combo_masks[k],
                                                    pred_rule=pred_rule)
                    targets.append((int(arm_bits[k]), float(y_sub == y_true)))
            for bits, r_obs in targets:
                j = bit_index[bits]
                combo_counts[j] += 1.0
                r_hat[j] += (r_obs - r_hat[j]) / combo_counts[j]

        # DUAL UPDATE -- identical to the EXP4 version.
        if not use_lp:
            # OMD dual ascent. Skipped under lp_chain/lp_full, where the LP
            # already enforces the budget exactly and lambda_t is its shadow
            # price rather than a quantity being learned.
            raw_lambda = lambda_t + step_size * (cost - remaining_training_budget_per_sample)
            lambda_t = max(0.0, min(lambda_max, raw_lambda))

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
        'warm_start': False,  # no expert weights to warm-start -- see docstring
        # greedy-specific extras
        'centers': est_centers,
        'combo_rewards': r_hat,
        'combo_counts': combo_counts,
        'est_counts': est_counts,
        'acquisition': acquisition,
        'n_arms': (len(combos) if combos is not None else 0),
        'avg_views_acquired': float(np.mean(views_trace)),
        'n_unique_masks': len(seen_masks),
    }