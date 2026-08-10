# -*- coding: utf-8 -*-
"""
core/submodular_greedy.py

Single canonical home for the MULTICLASS submodular greedy acquisition
oracle and the pairwise-Bhattacharyya risk it maximizes. Previously these
three functions lived inside gmm_submodular/gmm_multiclass_submodular.py;
they were MOVED here (not duplicated) the moment a second method family --
two_stage/two_stage_multiclass_greedy.py -- needed the same oracle.

=== Why moved rather than duplicated ===
This repo's usual convention is to DUPLICATE small helpers between
independent method families rather than cross-import (see
two_stage_multiclass.py's _macro_f1/_macro_ovr_auroc, which duplicate
gmm_multiclass_submodular.py's). That convention is deliberately BROKEN
here: two_stage_greedy exists specifically to answer "EXP4 vs. greedy,
everything else held fixed", and that comparison is only valid if both
methods run a bit-identical oracle. A copy that silently drifts would
invalidate the experiment, so there is exactly one copy and both callers
import it.

`multiclass_risk` here and `core.lp_colgen.multiclass_reward` are the SAME
formula (average pairwise Bhattacharyya accuracy proxy) under two names --
lp_colgen's copy is kept separate because it is consumed by the LP
branch-and-bound pricing routine and documented in that context. Both are
verbatim ports of the notebook's `multiclass_risk`.

=== Deliberate changes vs. the notebook (see git history of
    gmm_multiclass_submodular.py for the originals) ===
1. BUG FIX (carried over unchanged from gmm_multiclass_submodular.py's
   module docstring, item 1): the notebook computed `final_indices =
   free_indices + [best_single_item]` with free_indices a NUMPY ARRAY,
   which performs elementwise ADDITION (np.array([0]) + [5] ->
   array([5])), not list concatenation. free_indices is a Python list
   here, restoring the intended "free views + the one giant item"
   semantics.
2. NEW: `force_free` (default True) -- the free (zero-cost) views are
   ALWAYS included in the returned subset, instead of only when their
   marginal gain is strictly positive. See the parameter's docstring for
   why, and for why this is a no-op at alpha_ucb > 0.

=== Second move: greedy_chain / lp_policy_over_estimates / build_arm_tables
    (previously in two_stage/two_stage_multiclass_greedy.py) ===
Same reasoning as the first move, one level up. `greedy_chain` (the nested
action space) and `lp_policy_over_estimates` (the exact two-point solution
of the per-round budgeted LP over an ENUMERATED arm set) were written for
two_stage_greedy's center_update="reward_estimates" branch. They are now
also the acquisition policy behind
gmm_submodular/gmm_multiclass_submodular.run_training_phase's
acquisition="lp_chain" / "lp_full" modes, and the whole point of those
modes is "same action space and same LP, different learning setting", so
a drifting copy would invalidate the comparison exactly as a drifting
greedy_oracle would. two_stage_multiclass_greedy.py re-imports and
re-exports both names, so its existing call sites are unchanged.

`build_arm_tables` is NEW (not moved): it factors out the
bit_index / combo_masks / combo_cost / cost_order / arm_bits bookkeeping
that both callers need. two_stage_multiclass_greedy.py keeps its own
inline construction so that branch stays byte-for-byte as it was; the
helper is what the submodular caller uses. The one deliberate difference
is that build_arm_tables does NOT precompute the sub_arms containment
LIST -- that structure is O(n_arms^2) in memory and blows up on full
enumeration -- callers get `arm_bits` and do the O(n_arms) vectorised
`(arm_bits & played_bits) == arm_bits` test per round instead.
"""

from __future__ import annotations

import numba
import numpy as np

from core.lp_colgen import pairwise_diff_sq_from_means

# ─────────────────────────────────────────────────────────────────────────
# Shared policy vocabulary. CANONICAL HOME: both method families
# (gmm_submodular/gmm_multiclass_submodular.py and
# two_stage_greedy/two_stage_multiclass_greedy.py) re-export these rather
# than defining their own copies, because run_proposed_methods.py feeds ONE
# --acquisition / --reward-update flag to both and the two methods must
# agree on what the values mean. (Each module previously declared its own
# REWARD_UPDATE_SCOPES; the driver had to assert they had not drifted. One
# definition removes the need for the assert.)
# ─────────────────────────────────────────────────────────────────────────

# Which subset gets acquired each round. Identical meaning in both methods:
#   greedy    per-round submodular greedy oracle (greedy_oracle below)
#   lp_chain  per-round budgeted LP over the nviews+1 nested greedy chain
#   lp_full   the same LP over the full 2^(nviews-1) enumeration
ACQUISITION_MODES = ("greedy", "lp_chain", "lp_full")

# How the per-arm reward estimates are scored, under the lp_* modes only.
#   subsets   counterfactual replay of every arm contained in the played
#             set -- needs y_true, so this is FULL feedback
#   selected  the played arm only, from its 0/1 reward -- BANDIT feedback
REWARD_UPDATE_SCOPES = ("subsets", "selected")

# Eager enumeration is 2^(nviews-1) arms; the ceiling for acquisition="lp_full".
MAX_REWARD_ESTIMATE_VIEWS = 20


# ─────────────────────────────────────────────────────────────────────────
# Reward / risk -- verbatim notebook formulas (the notebook's "error rate"
# is actually the pairwise ACCURACY proxy 1 - exp(-0.125 * d), i.e. one
# minus the Bhattacharyya error bound; the name is kept for traceability).
# ─────────────────────────────────────────────────────────────────────────
@numba.njit
def bhattacharyya_error_rate(diff_mean_sq_mat):
    d_norm = np.sum(diff_mean_sq_mat, axis=0)  # (nc, nc)
    acc = np.exp(-0.125 * d_norm)  # 1/8 * |\delta\mu|^2
    return np.maximum(1.0 - acc, 0.0)


@numba.njit
def multiclass_risk(diff_mean_sq_mat):
    """Average pairwise Bhattacharyya accuracy proxy over the views
    included in diff_mean_sq_mat (axis 0). Monotone nondecreasing in the
    included view set, which is what makes the greedy below a (1-1/e)-type
    approximation and what the LP's branch-and-bound pruning bound relies
    on.

    NOTE (pre-existing, deliberately NOT "fixed"): the DIAGONAL of
    diff_mean_sq_mat is zero for true squared mean differences, but the
    callers pass an OPTIMISTIC tensor `diff_mean_sq + bonus` whose diagonal
    is 2/sqrt(count) > 0. So err_rate's diagonal is nonzero in practice and
    the sum below includes it. This is verbatim notebook behavior and it
    inflates every subset's score by a roughly common factor, so it does
    not change the argmax much -- but it is why absolute risk values are
    not directly interpretable as accuracies.
    """
    nc = diff_mean_sq_mat.shape[1]
    err_rate = bhattacharyya_error_rate(diff_mean_sq_mat)  # (nc, nc)
    denom = 1.0 / nc / (nc - 1)
    return 0.5 * denom * np.sum(err_rate)  # half: off-diagonal double-count


# ─────────────────────────────────────────────────────────────────────────
# Subset selection -- submodular greedy + giant-item check.
# ─────────────────────────────────────────────────────────────────────────
def greedy_oracle(diff_mean_sq, costs, omd_lambda, remain_budget,
                  free_indices, force_free=True):
    """
    Parameters
    ----------
    diff_mean_sq : (nviews, nc, nc)
        (Optimistic) squared pairwise per-view mean differences.
    costs : (nviews,) array
        Per-view acquisition costs. Zero-cost views are the "free" ones.
    omd_lambda : float
        Current dual variable / budget shadow price. A paid view is only
        added if its gain-per-unit-cost beats this.
    remain_budget : float
        Cap on the total cost this call may commit.
    free_indices : list[int]
        PYTHON LIST of zero-cost view indices (see module docstring, item 1).
    force_free : bool, default True
        Include every free view unconditionally, rather than only when its
        marginal gain is strictly positive.

        Rationale: every OTHER acquisition path in this repo already
        guarantees the free view is observed --
        `core.two_stage_utils.generate_view_combinations` prepends view 1
        to every EXP4 expert, and
        `core.lp_colgen.solve_lp_policy_colgen_multiclass` seeds
        free_mask[0]=True and only branches over PAID views. Without this
        flag the greedy oracle was the one path that could return a subset
        omitting the free view (or, if there were no free views at all, the
        EMPTY subset -> an empty x_obs at prediction time), which would mean
        two_stage and two_stage_greedy were being compared under
        different observation models.

        This is a NO-OP whenever the callers add a strictly positive
        optimism bonus (alpha_ucb > 0), since then every single-view
        marginal gain is strictly positive and the free views passed the
        old `margin_gain > 0` test anyway. It only bites at alpha_ucb == 0
        combined with a genuinely uninformative free view (identical class
        means on it). Set force_free=False to recover the exact pre-move
        behavior.

    Returns
    -------
    (nviews,) bool mask.
    """
    nview = len(costs)
    current_cost = 0.0
    current_objective = 0.0
    sel_set = []
    avail_elements = set(range(nview))

    # Harvest from free views first. `margin_gain` is defined as
    # risk(sel_set + [i]) - current_objective, so `current_objective +=
    # margin_gain` is an exact identity, not an approximation -- it stays
    # correct even when the gain is non-positive under force_free.
    for i in list(free_indices):
        tmp_select = sel_set + [i]
        margin_gain = multiclass_risk(diff_mean_sq[np.array(tmp_select)]) - current_objective
        if force_free or margin_gain > 0:
            sel_set.append(i)
            current_objective += margin_gain
            avail_elements.remove(i)

    copy_set = list(sel_set)
    while avail_elements and current_cost < remain_budget:
        best_margin = -1
        best_add = None
        for i in avail_elements:
            cost_i = costs[i]
            if current_cost + cost_i <= remain_budget:
                tmp_select = sel_set + [i]
                margin_gain = multiclass_risk(diff_mean_sq[np.array(tmp_select)]) - current_objective
                # no zero-cost elements remain here; epsilon for safety
                gain_ratio = margin_gain / (cost_i + 1e-9)
                # only add it if the margin beats the shadow price
                if gain_ratio > omd_lambda and gain_ratio > best_margin:
                    best_margin = gain_ratio
                    best_add = i
        if best_add is not None:
            sel_set.append(best_add)
            current_cost += costs[best_add]
            current_objective = multiclass_risk(diff_mean_sq[np.array(sel_set)])
            avail_elements.remove(best_add)
        else:
            break

    # Giant item check
    greedy_solution_reward = current_objective
    greedy_solution_indices = list(sel_set)
    best_single_item = None
    best_giant_reward = -np.inf
    for i in range(nview):
        if i in copy_set:
            continue
        cost_i = costs[i]
        if 0 < cost_i <= remain_budget:
            tmp_giant_indices = copy_set + [i]
            reward_with_giant = multiclass_risk(diff_mean_sq[np.array(tmp_giant_indices)])
            if reward_with_giant > best_giant_reward:
                best_giant_reward = reward_with_giant
                best_single_item = i

    if best_single_item is not None and best_giant_reward > greedy_solution_reward:
        # BUG FIX vs. the notebook: list concatenation, NOT numpy
        # elementwise addition -- see module docstring, item 1.
        final_indices = list(copy_set) + [best_single_item]
    else:
        final_indices = greedy_solution_indices

    mask = np.array([idx in final_indices for idx in range(nview)]).astype("bool")

    # Defensive: an all-False mask means an empty x_obs downstream. Under
    # force_free with at least one free view this is unreachable (every
    # repo dataset sets costs[0] == 0); the guard exists for the
    # force_free=False / no-free-view configurations.
    if not mask.any():
        mask[int(np.argmin(costs))] = True
    return mask

# ─────────────────────────────────────────────────────────────────────────
# Enumerated action spaces + the per-round LP over reward estimates.
# MOVED here from two_stage/two_stage_multiclass_greedy.py -- see the module
# docstring's "Second move" section for why these are shared rather than
# duplicated.
# ─────────────────────────────────────────────────────────────────────────
def greedy_chain(centers, costs, free_indices, force_free=True):
    """The FROZEN GREEDY CHAIN: the nested action space that replaces the
    full 2^(nviews-1) enumeration, cutting the arm count to nviews+1.

    Runs the same cost-benefit greedy as greedy_oracle above -- free views
    first, then repeatedly append the paid view with the best marginal-gain-
    per-unit-cost -- but with NO budget cap and NO lambda threshold,
    recording every intermediate set:

        S_0 = {free views} subset S_1 subset ... subset S_p = everything

    Returns
    -------
    list of 1-INDEXED view tuples ordered by increasing size, so callers'
    bit_index / combo_masks / combo_cost construction (and
    generate_view_combinations' output format) consume it with no changes.
    """
    nviews = centers.shape[1]
    diff_mean_sq = pairwise_diff_sq_from_means(centers)          # (nv, nc, nc)

    sel, objective = [], 0.0
    for i in list(free_indices):
        gain = multiclass_risk(diff_mean_sq[np.array(sel + [i])]) - objective
        if force_free or gain > 0:
            sel.append(i)
            objective += gain
    if not sel:            # no free view in this cost model
        sel = [int(np.argmin(costs))]
        objective = multiclass_risk(diff_mean_sq[np.array(sel)])

    chain = [sorted(sel)]                                        # S_0
    remaining = [i for i in range(nviews) if i not in sel]
    while remaining:
        best_ratio, best_add = -np.inf, None
        for i in remaining:
            gain = multiclass_risk(diff_mean_sq[np.array(sel + [i])]) - objective
            ratio = gain / (costs[i] + 1e-9)     # same 1e-9 as greedy_oracle
            if ratio > best_ratio:
                best_ratio, best_add = ratio, i
        sel.append(best_add)
        objective = multiclass_risk(diff_mean_sq[np.array(sel)])
        remaining.remove(best_add)
        chain.append(sorted(sel))                                # S_1 .. S_p

    chain.sort(key=len)
    return [tuple(v + 1 for v in s) for s in chain]              # 1-indexed


def lp_policy_over_estimates(ucb, combo_cost, cost_order, budget_per_round):
    """Exact optimum of the per-round budgeted LP over the ENUMERATED
    combinations:

        maximise    sum_c p_c * ucb_c
        subject to  sum_c p_c * cost_c <= budget_per_round
                    sum_c p_c = 1,  p >= 0

    One inequality plus the simplex means every basic feasible solution has
    at most TWO nonzeros, so the optimum is a mixture of two combinations
    and needs no LP solver: it is the upper concave envelope of the points
    (cost_c, ucb_c), evaluated at the budget.

    cost_order is an argsort of combo_cost computed ONCE at setup (costs
    never change), so this is a single O(n) pass per round.

    Returns (idx_lo, idx_hi, p_hi, lam):
        play idx_hi with probability p_hi, else idx_lo. `lam` is the slope
        of the active hull segment -- the LP's shadow price on the budget,
        i.e. exactly what the OMD dual was approximating, so callers report
        it as lambda / omd_lambda and keep Lagrangian traces comparable.
    """
    cs, us = combo_cost[cost_order], ucb[cost_order]
    # collapse duplicate costs to their best ucb, keeping the hull well posed
    pts, i, n = [], 0, len(cs)
    while i < n:
        j, best = i, i
        while j < n and cs[j] == cs[i]:
            if us[j] > us[best]:
                best = j
            j += 1
        pts.append((float(cs[i]), float(us[best]), int(cost_order[best])))
        i = j
    # upper hull, monotone chain over cost-ascending points
    hull = []
    for (x, y, idx) in pts:
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1):
                hull.pop()          # middle vertex is not above the chord
            else:
                break
        hull.append((x, y, idx))
    # beyond the max-ucb vertex the envelope only falls; spending more can
    # never help, which is how the "<=" in the budget constraint is honoured
    kbest = max(range(len(hull)), key=lambda k: hull[k][1])
    hull = hull[:kbest + 1]

    b = float(budget_per_round)
    if b <= hull[0][0] or len(hull) == 1:
        return hull[0][2], hull[0][2], 0.0, 0.0
    if b >= hull[-1][0]:
        # budget not binding: play the best combination outright
        return hull[-1][2], hull[-1][2], 0.0, 0.0
    lo = 0
    while lo + 1 < len(hull) and hull[lo + 1][0] <= b:
        lo += 1
    (x1, y1, i1), (x2, y2, i2) = hull[lo], hull[lo + 1]
    span = x2 - x1
    p_hi = 0.0 if span <= 0 else (b - x1) / span
    lam = 0.0 if span <= 0 else (y2 - y1) / span
    return i1, i2, float(min(max(p_hi, 0.0), 1.0)), float(max(lam, 0.0))


def build_arm_tables(combos, costs, nviews):
    """Bookkeeping shared by every enumerated-action-space caller.

    Parameters
    ----------
    combos : list of 1-INDEXED view tuples
        Either greedy_chain(...) or
        core.two_stage_utils.generate_view_combinations(nviews).
    costs : (nviews,) array, 0-indexed per-view costs.

    Returns
    -------
    dict with
        combo_masks : (n_arms, nviews) bool
        combo_cost  : (n_arms,) float   -- total cost of each arm
        cost_order  : (n_arms,) int     -- stable argsort of combo_cost,
                                           computed once (costs never change)
        arm_bits    : (n_arms,) int64   -- bitmask of each arm, arm j's bit i
                                           set iff view i is in arm j
        bit_index   : dict bitmask -> arm index

    Deliberately does NOT build the sub_arms containment list: it is
    O(n_arms^2) memory, which is fine for the nviews+1 chain and fatal for
    the 2^(nviews-1) enumeration. Callers get arm_bits and test containment
    with the vectorised `(arm_bits & played_bits) == arm_bits`.
    """
    n_arms = len(combos)
    combo_masks = np.zeros((n_arms, nviews), dtype=bool)
    combo_cost = np.zeros(n_arms, dtype=np.float64)
    arm_bits = np.zeros(n_arms, dtype=np.int64)
    bit_index = {}
    for j, combo in enumerate(combos):
        idx = np.asarray(combo, dtype=int) - 1
        combo_masks[j, idx] = True
        combo_cost[j] = float(costs[combo_masks[j]].sum())
        bits = int(sum(1 << int(i) for i in idx))
        arm_bits[j] = bits
        bit_index[bits] = j
    return {
        "combo_masks": combo_masks,
        "combo_cost": combo_cost,
        "cost_order": np.argsort(combo_cost, kind="stable"),
        "arm_bits": arm_bits,
        "bit_index": bit_index,
    }


def mask_to_bits(mask):
    """0-indexed boolean view mask -> integer bitmask (bit i = view i)."""
    return int(sum(1 << i for i in range(len(mask)) if mask[i]))