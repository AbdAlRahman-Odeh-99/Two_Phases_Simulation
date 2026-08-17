# -*- coding: utf-8 -*-
"""
core/submodular_greedy.py

Single canonical home for the MULTICLASS submodular greedy acquisition
oracle and the pairwise-Bhattacharyya risk it maximizes. Previously these
three functions lived inside gmm_submodular/gmm_multiclass_submodular.py;
they were MOVED here (not duplicated) the moment a second method family --
two_stage/two_stage_multiclass_greedy.py -- needed the same oracle.

=== Instrumentation note ===
The five acquisition entry points below carry a
@timed("t_acquisition") decorator from core.logging_utils. That single
placement is why the new t_acquisition column is populated for BOTH method
families and for EVERY acquisition mode without either method module being
edited: both of them reach their per-round subset choice through exactly
these functions. Decorators, not `with` blocks, so no numerical body was
re-indented. The decorator is a no-op when timing is disabled and when no
runner is collecting, so importing this module standalone costs nothing.
Do NOT add a t_acquisition tick at any CALLER of these -- ticks accumulate
per name, and a caller-side tick around a call to a decorated function
would count the same span twice.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.optimize as opt

from core.logging_utils import bump, timed

ACQUISITION_MODES = ("greedy", "lp_chain", "lp_full", "lp_full_opt", "ucb_argmax", "hedge")
LP_ACQUISITION_MODES = ("lp_chain", "lp_full", "lp_full_opt")
ORACLE_ACQUISITION_MODES = ("lp_full_opt",)
ARGMAX_ACQUISITION_MODES = ("ucb_argmax",)
HEDGE_ACQUISITION_MODES = ("hedge",)
FULL_ENUMERATION_MODES = ("lp_full", "lp_full_opt", "ucb_argmax", "hedge")
REWARD_UPDATE_SCOPES = ("subsets", "selected")
MAX_REWARD_ESTIMATE_VIEWS = 20

# === reward_estimate: WHAT the arm values are, not whether they are unbiased
#
# RENAMED. These were "biased" / "unbiased", which collided head-on with the
# OTHER "unbiased" in this codebase -- the containment-replay acquisition
# ported from multiclass_supervised_unbiased_adaptive.ipynb, whose
# `sim_unbiased` is about REPLAY SCOPE (score every contained subset from
# one observed row), not about estimator bias. Two unrelated meanings on
# one word, one of which was never accurate anyway: the old "unbiased"
# table is a uniform running mean over rounds whose classifier keeps
# improving, so it lags and is biased low (see the CAVEAT in both method
# modules). The names now say what the values ARE:
#
#   "surrogate" (was "biased")    the closed-form Bhattacharyya proxy,
#                                 computed from estimated means, needs no
#                                 observations and no enumeration.
#   "empirical" (was "unbiased")  a measured per-subset accuracy table,
#                                 filled in from actual predictions, needs
#                                 2^(nviews-1) arms.
REWARD_ESTIMATES = ("surrogate", "empirical")


def validate_reward_estimate(reward_estimate):
    """Membership check for reward_estimate. Returns it unchanged.

    A function rather than an inline `not in` test only so the message is
    written once for the three call sites (both method modules and the
    unified runner).
    """
    if reward_estimate not in REWARD_ESTIMATES:
        raise ValueError(
            f"reward_estimate must be one of {REWARD_ESTIMATES}; got "
            f"{reward_estimate!r}. NOTE these were renamed from "
            f"'biased'/'unbiased' -- 'biased' is now 'surrogate' and "
            f"'unbiased' is now 'empirical'. The old names are NOT "
            f"accepted; update the call site or script.")
    return reward_estimate


def uses_empirical_arm_rewards(acquisition, reward_estimate="surrogate"):
    return (acquisition in ("lp_chain", "lp_full", "ucb_argmax", "hedge") or (acquisition == "greedy" and reward_estimate == "empirical"))

# Reward
@numba.njit
def bhattacharyya_accuracy_proxy(diff_mean_sq_mat):
    d_norm = np.sum(diff_mean_sq_mat, axis=0)  # (nc, nc)
    # Bhattacharyya overlap-based proxy for classification error
    error = np.exp(-0.125 * d_norm)  # 1/8 * |\delta\mu|^2
    return np.maximum(1.0 - error, 0.0)
@numba.njit
def multiclass_reward(diff_mean_sq_mat):
    nc = diff_mean_sq_mat.shape[1]
    pairwise_acc = bhattacharyya_accuracy_proxy(diff_mean_sq_mat)  # (nc, nc)
    denom = 1.0 / nc / (nc - 1)
    return 0.5 * denom * np.sum(pairwise_acc)
def pairwise_diff_sq_from_means(est_means):
    mean_tr = np.asarray(est_means, dtype=np.float64).T  # (nviews, nc)
    w_diff = mean_tr[:, :, None] - mean_tr[:, None, :]   # (nviews, nc, nc)
    return np.square(w_diff)


# Subset selection
@timed("t_acquisition")
def greedy_oracle(diff_mean_sq, costs, omd_lambda, remain_budget,
                  free_indices, force_free=True,
                  gain_func=None, empty_value=0.0):
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

    Returns
    -------
    (nviews,) bool mask.
    """
    if gain_func is None:
        gain_func = lambda sel: multiclass_reward(diff_mean_sq[np.array(sel)])
    nview = len(costs)
    current_cost = 0.0
    # Value of the EMPTY set. 0.0 is right for the Bhattacharyya surrogate
    # (d=0 -> 1-exp(0)=0) but WRONG for a learned accuracy table, where the
    # empty set scores chance = 1/nclasses. Leaving it at 0 inflates every
    # first-view marginal gain and the ratio test passes for anything.
    current_objective = empty_value
    sel_set = []
    avail_elements = set(range(nview))
    # Harvest from free views first. `margin_gain` is defined as
    # risk(sel_set + [i]) - current_objective
    for i in list(free_indices):
        tmp_select = sel_set + [i]
        margin_gain = gain_func(tmp_select) - current_objective
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
                margin_gain = gain_func(tmp_select) - current_objective
                # no zero-cost elements remain here; epsilon for safety
                gain_ratio = margin_gain / (cost_i + 1e-9)
                # only add it if the margin beats the shadow price
                if gain_ratio > omd_lambda and gain_ratio > best_margin:
                    best_margin = gain_ratio
                    best_add = i
        if best_add is not None:
            sel_set.append(best_add)
            current_cost += costs[best_add]
            current_objective = gain_func(sel_set)
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
            reward_with_giant = gain_func(tmp_giant_indices)
            if reward_with_giant > best_giant_reward:
                best_giant_reward = reward_with_giant
                best_single_item = i
    if best_single_item is not None and best_giant_reward > greedy_solution_reward:
        final_indices = list(copy_set) + [best_single_item]
    else:
        final_indices = greedy_solution_indices
    mask = np.array([idx in final_indices for idx in range(nview)]).astype("bool")
    if not mask.any():
        mask[int(np.argmin(costs))] = True
    return mask

# ─────────────────────────────────────────────────────────────────────────
# Enumerated action spaces + the per-round LP over reward estimates.
# ─────────────────────────────────────────────────────────────────────────
@timed("t_acquisition")
def greedy_chain(centers, costs, free_indices, force_free=True,
                 gain_func=None, empty_value=0.0):
    """The GREEDY CHAIN: the nested action space that replaces the
    full 2^(nviews-1) enumeration, cutting the arm count to nviews+1.

    Runs the same cost-benefit greedy as greedy_oracle above -- free views
    first, then repeatedly append the paid view with the best marginal-gain-
    per-unit-cost:
        S_0 = {free views} subset S_1 subset ... subset S_p = everything
    Returns
    -------
    list of 1-INDEXED view tuples ordered by increasing size, so callers'
    bit_index / combo_masks / combo_cost construction (and
    generate_view_combinations' output format) consume it with no changes.
    """
    nviews = centers.shape[1]
    if gain_func is None:
        diff_mean_sq = pairwise_diff_sq_from_means(centers)  # (nv, nc, nc)
        gain_func = lambda sel: multiclass_reward(diff_mean_sq[np.array(sel)])
    sel, objective = [], empty_value
    for i in list(free_indices):
        gain = gain_func(sel + [i]) - objective
        if force_free or gain > 0:
            sel.append(i)
            objective += gain
    if not sel:            # no free view in this cost model
        sel = [int(np.argmin(costs))]
        objective = gain_func(sel)
    chain = [sorted(sel)]                                        # S_0
    remaining = [i for i in range(nviews) if i not in sel]
    while remaining:
        best_ratio, best_add = -np.inf, None
        for i in remaining:
            gain = gain_func(sel + [i]) - objective
            ratio = gain / (costs[i] + 1e-9)     # same 1e-9 as greedy_oracle
            if ratio > best_ratio:
                best_ratio, best_add = ratio, i
        sel.append(best_add)
        objective = gain_func(sel)
        remaining.remove(best_add)
        chain.append(sorted(sel))                                # S_1 .. S_p
    chain.sort(key=len)
    return [tuple(v + 1 for v in s) for s in chain]              # 1-indexed

@timed("t_acquisition")
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
    bump("n_lp_solves")
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

@timed("t_acquisition")
def linprog_policy_over_estimates(ucb, combo_cost, budget_per_round):
    """Exact optimum of the per-round budgeted LP over the ENUMERATED
    combinations, solved with a general LP solver:

        maximise    sum_c p_c * ucb_c
        subject to  sum_c p_c * cost_c <= budget_per_round
                    sum_c p_c = 1,  p >= 0

    Returns (p, lam):
        p    (n_combos,) probability vector -- SAMPLE from it. The previous
             implementation returned a two-point mixture (idx_lo, idx_hi,
             p_hi) because one inequality plus the simplex means SOME
             optimal vertex has at most two nonzeros; the solver may
             instead land on an interior optimal face when arms tie, so the
             full vector is returned and callers draw from it.
        lam  shadow price on the budget constraint, >= 0, taken from the
             LP dual rather than a hull slope. Same meaning as before --
             what the OMD dual was approximating -- so Lagrangian traces
             stay comparable across acquisition modes.
    """
    bump("n_lp_solves")
    ucb = np.asarray(ucb, dtype=np.float64).ravel()
    combo_cost = np.asarray(combo_cost, dtype=np.float64).ravel()
    n = ucb.shape[0]
    b = max(0.0, float(budget_per_round))

    res = opt.linprog(
        -ucb,
        A_ub=combo_cost.reshape(1, -1),
        b_ub=np.array([b]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=(0.0, 1.0),
        method="highs",
    )

    if not res.success:
        # Unreachable under this codebase's cost convention (the free-view
        # arm has cost 0, so p = e_free is feasible at any b >= 0), but fall
        # back to the cheapest arm explicitly rather than returning garbage.
        #
        # LOGGED, not silent: this branch used to be reachable only in
        # theory, and if it ever does fire the run keeps going with a
        # degenerate policy that looks like a legitimate result. A WARNING
        # is the difference between noticing that and not.
        from core.logging_utils import get_logger
        get_logger("afa.acquisition").warning(
            "linprog FAILED (status=%s, %s) at budget_per_round=%.6g over %d arms; "
            "falling back to the cheapest arm for this round",
            getattr(res, "status", "?"), getattr(res, "message", ""), b, n)
        p = np.zeros(n)
        p[int(np.argmin(combo_cost))] = 1.0
        return p, 0.0

    p = np.clip(res.x, 0.0, None)
    total = p.sum()
    if total <= 0:
        p = np.zeros(n)
        p[int(np.argmin(combo_cost))] = 1.0
    else:
        p = p / total          # exact simplex sum, for rng.choice

    # HiGHS reports d(objective)/d(b_ub) for the MINIMISED objective -ucb@p,
    # so the marginal is <= 0 and its negation is the gain in expected ucb
    # per unit of budget -- matching the old hull slope's sign convention.
    lam = 0.0
    marg = getattr(res, "ineqlin", None)
    if marg is not None and len(marg.marginals):
        lam = -float(marg.marginals[0])

    return p, max(lam, 0.0)

@timed("t_acquisition")
def argmax_policy_over_estimates(ucb, combo_cost, omd_lambda,
                                 remain_budget=None):
    """The NOTEBOOK's acquisition rule (acquisition="ucb_argmax"): pick the
    single arm maximising the Lagrangian

        score_c = ucb_c - omd_lambda * cost_c

    over the ENUMERATED arms, with no randomisation and no LP.

    Differences from lp_policy/linprog_policy_over_estimates, which share
    this call site's inputs:

      * DETERMINISTIC. The LP returns a distribution and the caller draws
        from it, so the LP can hit a per-round budget exactly by mixing two
        arms. This returns one index; the budget is respected only in
        expectation, through omd_lambda, exactly as acquisition="greedy"
        does. The caller must therefore keep running its OMD dual update --
        with omd_lambda pinned at 0 this degenerates to "always buy the
        highest-ucb arm", which is usually the full view set.
      * COST ENTERS LINEARLY, not as a constraint. An arm whose cost exceeds
        what is left is still scored; `remain_budget` (optional) filters it
        out afterwards.

    Parameters
    ----------
    ucb : (n_arms,) array
        Arm values, already including any exploration bonus. NOT capped at
        1.0 here, deliberately -- see the CAPPING note in the two callers'
        module docstrings: capping an argmax collapses saturated arms into a
        tie that np.argmax always breaks toward index 0.
    combo_cost : (n_arms,) array, total cost of each arm.
    omd_lambda : float, current dual variable / shadow price.
    remain_budget : float or None
        If given, arms costing more than this are excluded. When nothing is
        affordable the cheapest arm is returned rather than raising -- with
        this codebase's cost convention (arm 0 is the free view alone, cost
        0) that is the free-only fallback every other mode also uses.

    Returns
    -------
    int : index into `ucb` / `combo_cost`.
    """
    ucb = np.asarray(ucb, dtype=np.float64).ravel()
    combo_cost = np.asarray(combo_cost, dtype=np.float64).ravel()
    score = ucb - float(omd_lambda) * combo_cost
    if remain_budget is not None:
        affordable = combo_cost <= float(remain_budget) + 1e-12
        if not affordable.any():
            return int(np.argmin(combo_cost))
        score = np.where(affordable, score, -np.inf)
    return int(np.argmax(score))


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


def arm_accuracies_from_means(X_rows, Y_rows, means, combo_masks):
    """Per-arm EMPIRICAL nearest-centroid accuracy of `means`, restricted to
    each arm's view set, measured on (X_rows, Y_rows).

    This is the arm-value function for the ORACLE acquisition modes: pass
    the TRUE generative means and the rows the policy is about to run over,
    and the result is each subset's exact achievable accuracy -- the same
    quantity r_hat is a running estimate of under lp_chain / lp_full, on the
    same 0/1 scale, so an oracle run and a learning run are directly
    comparable arm for arm.

    Returns (n_arms,) float. An empty row set yields a flat chance-level
    1/nclasses, which keeps the LP well posed (it then just buys the
    cheapest arm) instead of raising inside the solver.
    """
    combo_masks = np.asarray(combo_masks, dtype=bool)
    means = np.asarray(means, dtype=np.float64)
    n_arms = combo_masks.shape[0]
    nclasses = means.shape[0]

    xs = np.asarray(X_rows, dtype=np.float64)
    ys = np.asarray(Y_rows, dtype=int)
    if len(xs) == 0:
        return np.full(n_arms, 1.0 / nclasses, dtype=np.float64)
    if means.shape[1] != combo_masks.shape[1]:
        raise ValueError(
            f"means has {means.shape[1]} views but combo_masks has "
            f"{combo_masks.shape[1]}. Pass true means at the POST-truncation "
            f"width -- see core.optimal_static.synthetic_true_means' "
            f"n_views_used parameter, which exists for exactly this mismatch.")

    acc = np.empty(n_arms, dtype=np.float64)
    for j in range(n_arms):
        m = combo_masks[j]
        d = ((xs[:, None, m] - means[None, :, m]) ** 2).sum(axis=2)  # (n, nc)
        acc[j] = float(np.mean(d.argmin(axis=1) == ys))
    return acc