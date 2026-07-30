# -*- coding: utf-8 -*-
"""
two_stage_multiclass_greedy.py

Sibling of two_stage_multiclass.py that replaces ONE thing: the Stage-2
TRAINING acquisition policy. Everything else -- Stage 1's supervised centre
init, the prediction rules, the metrics, the budget accounting conventions,
and Stage 2's LP-column-generation INFERENCE -- is imported UNCHANGED from
two_stage_multiclass.py, so the two files differ only in how a view subset
is chosen each training round:

    two_stage_multiclass.run_alg_multiclass
        EXP4/Hedge-BwK bandit sampling over all 2^(nviews-1) view-
        combination "experts", + OMD dual.
    two_stage_multiclass_greedy.run_alg_greedy_multiclass   (THIS FILE)
        per-round submodular greedy over the optimistic pairwise-
        Bhattacharyya risk (core.submodular_greedy.greedy_oracle), the
        acquisition rule gmm_multiclass_submodular.py's training phase uses,
        + the SAME OMD dual.

Kept as a NEW SIBLING FILE rather than a flag on two_stage_multiclass.py,
following the same convention as two_stage_multiclass.py vs. the deleted
two_stage_asymmetric.py, and gmm_multiclass_submodular.py vs.
gmm_2class_submodular_asymmetric.py: both stay independently runnable and
comparable, and neither file's numbers can shift because of a change made
for the other.

=== What this buys ===
1. No expert enumeration. n_experts = 2^(nviews-1) is what forces
   two_stage to keep --max-modalities <= MAX_RECOMMENDED_MODALITIES and
   --image-pool-side tiny. The greedy oracle is O(nviews^2) risk
   evaluations per round, so this variant runs at ANY nviews the LP
   inference can still handle (i.e. the same regime as submodular).
2. A controlled EXP4-vs-greedy comparison. Same Stage-1 centres, same
   budget split, same prediction rule, same OMD dual update, same LP
   inference, same metrics -- the acquisition policy is the only moving
   part (in center_update="frozen"/"counts"; see below).

=== center_update: what happens to the centres during Stage 2 ===
two_stage's premise is that the CLASSIFIER is frozen after Stage 1 --
run_alg_multiclass never touches the centres. But greedy_oracle needs a
reward model, and that model is a plug-in of the centres, so "freeze
everything" freezes the ACQUISITION policy too. The modes below separate
the two: which centres the CLASSIFIER uses, and which centres the
ACQUISITION reward estimate uses.

  "frozen"        Both frozen. Fully static oracle -- it re-derives one
                  subset per lambda and never learns. Purest baseline.
  "reward_full"   (DEFAULT) TWO sets of centres. Prediction keeps the
                  Stage-1 centres untouched; a SEPARATE acq_centers
                  estimate, initialised from them, is updated each round
                  from the revealed label and feeds greedy_oracle. The
                  deployed classifier is committed after Stage 1, but the
                  measurement policy keeps learning.
                  NOTE: the reward is a deterministic plug-in of the means
                  (d = (mu_k - mu_l)^2), so "an updated reward estimate" IS
                  "an updated second set of means" -- no estimator of d
                  from labelled samples avoids factoring through per-class
                  means.
  "reward_bandit" Same split, but acq_centers is updated from the 1-bit
                  reward (complementary-label step on a mistake).
  "full"          centres AND counts updated from the revealed true label, i.e.
                  gmm_multiclass_submodular.run_training_phase's
                  feedback="full" update, verbatim. Prediction and
                  acquisition share a single estimate again.
  "bandit"        centres AND counts updated from the 1-bit reward, i.e.
                  that function's feedback="bandit" update, verbatim.

*** MEASURED: the reward_* modes buy nothing ***
reward_full is statistically indistinguishable from a frozen acquisition
model (20 seeds x K in {4,10} x 2 budget fractions: mean delta -0.0000,
p = 0.99); reward_bandit is WORSE (-0.0146, p = 3e-10). NOT because the
update is inert -- reward_full picks a DIFFERENT subset from "frozen" in
79.8% of rounds, the same divergence rate as "full". The reshuffled choices
are simply equally good: with Stage-1 centres the RANKING of views by
discriminability is already good enough, and many similar-cost subsets are
near-equally informative. All of "full"'s advantage (+0.0432 over
reward_full, p = 1e-23, 78/80 seeds) comes from the CLASSIFIER improving,
not from better acquisition. Keep the reward_* modes as the ablation that
establishes this; do not expect them to win.

*** Comparability warning for "full"/"bandit" ***
Those two modes make this method "submodular with a supervised
warm-start phase", not "two_stage with a different acquisition rule": the
centres are no longer the Stage-1 centres, and "full" additionally reveals
y_true every round, which the EXP4 baseline never sees (run_alg_multiclass
observes ONLY reward = float(y_hat == y[t])). They are worth running -- at
init_fraction == 0 they should reduce to submodular almost exactly,
which is the natural sanity check -- but a "full" row is NOT an
apples-to-apples comparison against a two_stage row. "counts" is.
That is why "counts" increments est_counts[y_pred] rather than
est_counts[y_true]: two_stage's Stage 2 is bandit-feedback by
construction, and keying the counts off y_true would leak label
information into the acquisition policy through the back door.

=== There is deliberately NO warmup ===
gmm_multiclass_submodular.run_training_phase opens with `nclasses` rounds
that acquire ALL views and predict uniformly at random, so est_means has
real observations before the greedy oracle is asked to trust it.

two_stage ALREADY HAS THAT, in a strictly better form: Stage 1
(initialize_centers_multiclass) observes all views for T1 rounds and does a
SUPERVISED running-mean update on the TRUE class. Same mechanism, tunable
number of rounds, and no random predictions polluting the score. Adding a
warmup on top of Stage 1 duplicates it, and duplicates it badly:

  - center_update="frozen"/"counts" never touch the centres, so a warmup
    acquires every view and then discards the data outright, leaving only
    `nclasses` random predictions on the scoreboard.
  - center_update="full"/"bandit" would consume it, but Stage 1 already ran.

An earlier version of this file had a `warmup` parameter. REMOVED after
measurement: at K=10, nviews=10, 15 seeds, budget_fraction 0.3 it cost
0.019 (frozen) / 0.027 (counts) / 0.040 (full) accuracy versus no warmup,
overall p<0.001; at K=4 the two were statistically indistinguishable. Even
at T1 == 0 -- the only case with no Stage 1 to duplicate -- no-warmup was
slightly ahead. Its name also invited confusion with the EXP4 sibling's
`warm_start`, an unrelated mechanism that primes the expert WEIGHT VECTOR
from a closed-form prior at zero acquisition cost.

The real "move Stage-1 information into the acquisition policy for free"
mechanism here is stage1_counts() below: it seeds est_counts so the
optimism bonus starts calibrated instead of at 1 everywhere. That is the
greedy analogue of warm_start -- no budget, no random predictions,
unconditional.

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
from core.submodular_greedy import greedy_oracle

# Everything class-count-agnostic and policy-independent is reused
# UNCHANGED from the EXP4 sibling -- same method family, so a cross-import
# is correct here (and required: the two variants MUST share these to be
# comparable).
from two_stage.two_stage_multiclass import (
    PRED_RULES,                                  # noqa: F401 -- re-exported
    _class_posterior_scores,
    _macro_f1,
    _macro_ovr_auroc,
    _pred_nearest_center,
    _pred_pairwise_vote,
    calculate_two_stage_error,                   # noqa: F401 -- re-exported
    initialize_centers_multiclass,               # noqa: F401 -- re-exported
    run_inference_lp_colgen_multiclass,          # noqa: F401 -- re-exported
    run_inference_lp_dataset_colgen_multiclass,  # noqa: F401 -- re-exported
)

CENTER_UPDATE_MODES = ("frozen", "reward_full", "reward_bandit",
                       "reward_estimates", "full", "bandit")
REWARD_UPDATE_SCOPES = ("subsets", "selected")

# reward_estimates enumerates all 2^(nviews-1) combinations EAGERLY, so it
# carries two_stage's action-space ceiling; the other modes do not.
MAX_REWARD_ESTIMATE_VIEWS = 20


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

    ---- The closed-form alternative, deliberately NOT used ----
    two_stage's EXP4 warm start priors its experts with
    combo_closed_form_reward_multiclass(): a plug-in computed from the
    centres that never looks at data. It is kept here, commented out,
    because it is a good ORDERING prior but a bad VALUE prior -- and UCB
    arithmetic is on values. Measured at nviews=6, T1=100, the closed form
    spans 0.010-0.146 while true accuracy spans 0.340-0.600 (Spearman
    0.90); seeding UCB with the former and updating it with the latter
    leaves every arm looking permanently starved. To switch, replace the
    body below with:

        from two_stage.two_stage_multiclass import (
            combo_closed_form_reward_multiclass)
        r_hat = np.array([combo_closed_form_reward_multiclass(centers, c, nviews)
                          for c in combos], dtype=np.float64)
        counts = np.ones(len(combos), dtype=np.float64)   # no real observations
        return r_hat, counts
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
# RETIRED: greedy subset construction over the reward estimates.
#
# reward_estimates now acquires via lp_policy_over_estimates() below. This
# is kept, commented out, because it was the first cut and is the natural
# thing to reach for -- but it was never justified in this setting:
#   * its gains are differences of two NOISY empirical estimates, so the
#     objective is not submodular, the (1-1/e) guarantee does not hold, and
#     greedy_oracle's giant-item check had to be dropped;
#   * it is deterministic, so the arms it evaluates are self-selected and
#     the estimates drift toward whatever it already prefers;
#   * it approximates, via OMD dual ascent on lambda, the very LP that the
#     hull below solves exactly and in closed form.
# To restore it, uncomment and swap the call site in the selection branch.
# ─────────────────────────────────────────────────────────────────────────
# def greedy_over_estimates(ucb, bit_index, per_view_costs, start_bits,
#                           paid_views, omd_lambda, remain_budget):
#     """Incremental subset build scored by the per-combination reward
#     ESTIMATE, instead of greedy_oracle's per-view Bhattacharyya
#     decomposition.
#
#     Same skeleton as greedy_oracle: start from the free views, repeatedly
#     add the paid view with the best gain-per-unit-cost, stop when nothing
#     beats the shadow price omd_lambda or the budget is exhausted.
#
#     IMPORTANT DIFFERENCE: these gains are differences of two NOISY empirical
#     estimates, so the objective is NOT submodular. The (1-1/e) guarantee
#     does not hold, and greedy_oracle's giant-item check has no rationale
#     here, so it is deliberately omitted. This is a heuristic search over the
#     estimates, not an approximation algorithm.
#     """
#     cur_bits = start_bits
#     cur_score = ucb[bit_index[cur_bits]]
#     spent = 0.0
#     while True:
#         best = None
#         for v in paid_views:
#             bit = 1 << v
#             if cur_bits & bit:
#                 continue
#             c_v = per_view_costs[v]
#             if spent + c_v > remain_budget:
#                 continue
#             cand_idx = bit_index[cur_bits | bit]
#             ratio = (ucb[cand_idx] - cur_score) / (c_v + 1e-9)
#             if ratio > omd_lambda and (best is None or ratio > best[0]):
#                 best = (ratio, v, cand_idx)
#         if best is None:
#             break
#         _, v, cand_idx = best
#         cur_bits |= 1 << v
#         cur_score = ucb[cand_idx]
#         spent += per_view_costs[v]
#     return cur_bits


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
        i.e. exactly what the OMD dual was approximating, so it is reported
        as lambda_t and keeps avg_lagrangian_reward comparable.
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


def _submasks_containing(bits, required_bits):
    """Every sub-bitmask of `bits` that still contains `required_bits`."""
    out, sub = [], bits
    while True:
        if (sub & required_bits) == required_bits:
            out.append(sub)
        if sub == 0:
            break
        sub = (sub - 1) & bits
    return out


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 (training phase): per-round submodular greedy + OMD dual.
#
# The dual update, the budget bookkeeping, the reward definition, the
# Lagrangian trace and the returned dict are BYTE-FOR-BYTE the same logic
# as two_stage_multiclass.run_alg_multiclass -- only the block that chooses
# `mask` (and the optional centre/count update) differs.
# ─────────────────────────────────────────────────────────────────────────
def run_alg_greedy_multiclass(x, y, centers, costs, T1, training_budget, rng,
                              center_update="reward_full", alpha_ucb=2.0, lr=1e-2,
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
        Stage-1 learned centres. NEVER mutated in place -- an internal copy
        is updated (when center_update says so) and returned under the
        'centers' key, so the caller decides what Stage-2 inference uses.
    costs : (nviews,) array
        Per-view costs, 0-indexed, free views at cost 0. This REPLACES the
        EXP4 version's (view_combinations, combo_costs) pair.
    T1, training_budget, rng, step_size, lambda_max, pred_rule
        Identical meaning to run_alg_multiclass.
    center_update : {"frozen", "counts", "full", "bandit"}
        See the module docstring. Default "counts".
    alpha_ucb : float, default 2.0
        Optimism scale in the bonus sqrt(alpha_ucb * log(t+1)) / sqrt(count)
        -- same knob and same default as
        gmm_multiclass_submodular.run_training_phase. alpha_ucb=0 makes
        the oracle purely exploitative (and, under center_update="frozen",
        fully deterministic given lambda).
    lr : float, default 1e-2
        Complementary-label gradient step, center_update="bandit" only.
    force_free : bool, default True
        Passed through to greedy_oracle -- keeps the free view(s) in every
        acquired subset, matching EXP4's and the LP's invariant.

    Returns
    -------
    dict with the SAME keys run_alg_multiclass returns (so the runner and
    run_proposed_methods.normalize_two_stage_mc need no changes), plus
    'centers', 'est_counts', 'center_update', 'avg_views_acquired',
    'n_unique_masks'. 'warm_start' is always False -- there is no expert
    weight vector to warm-start here; the Stage-1 information enters
    through the centres and (via stage1_counts) the optimism bonus instead.
    """
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")
    if center_update not in CENTER_UPDATE_MODES:
        raise ValueError(f"center_update must be one of {CENTER_UPDATE_MODES}, "
                         f"got {center_update!r}")

    total_samples = len(x)
    T2 = total_samples - T1
    if T2 <= 0:
        return {}

    costs = np.asarray(costs, dtype=np.float64)
    nviews = centers.shape[1]
    nclasses = centers.shape[0]

    # Never mutate the caller's array: the runner reuses `centers` for
    # two_stage_error's init_error and (in frozen/counts mode) for
    # inference. Update modes hand the updated copy back instead.
    est_centers = np.array(centers, dtype=np.float64, copy=True)
    # SEPARATE acquisition-side estimate, identical to est_centers at the
    # start. Under reward_full/reward_bandit this is the array that moves
    # while est_centers (the CLASSIFIER) stays exactly as Stage 1 left it.
    # Under full/bandit the two are re-aliased so there is one estimate.
    acq_centers = np.array(centers, dtype=np.float64, copy=True)
    est_counts = stage1_counts(y, T1, nclasses, nviews)
    one_vec = np.ones(nclasses)

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
    # ── reward_estimates setup (EAGER: all 2^(nviews-1) combinations) ──
    combos = bit_index = r_hat = combo_counts = None
    combo_masks = combo_cost = cost_order = None
    start_bits = 0
    if center_update == "reward_estimates":
        if nviews > MAX_REWARD_ESTIMATE_VIEWS:
            raise ValueError(
                f"center_update='reward_estimates' enumerates 2^(nviews-1) = "
                f"2^{nviews - 1} combinations eagerly; nviews={nviews} exceeds "
                f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. Use "
                f"--max-modalities to trim, or a mode that does not enumerate "
                f"(frozen / reward_full / full).")
        combos = generate_view_combinations(nviews)
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
            # generate_view_combinations only forces view 1; if some other
            # view is also free, fall back to view 1 alone as the seed.
            start_bits = 1
    lambda_t = 0.0

    predictions = []
    score_rows = []
    true_labels = []
    reward_trace = []
    lagrangian_reward_trace = []
    views_trace = []
    seen_masks = set()
    errors = 0

    for t in range(T1, total_samples):
        round_idx = t - T1  # 0-based Stage-2 round, drives the log(t+1) bonus

        if training_remaining_budget <= 0:
            # SAFE BUDGET CHECK (BwK-style absorbing fallback policy) --
            # same semantics as run_alg_multiclass's cheapest-combo branch.
            mask = free_only_mask.copy()
        elif center_update == "reward_estimates":
            # UCB on the per-combination AVERAGE ACCURACY (the real
            # objective), not on the Bhattacharyya surrogate.
            ucb = r_hat + np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts)
            # Per-round budget target, RECOMPUTED each round so it
            # self-corrects for over/underspend (unlike the fixed target the
            # OMD dual chases in the other modes).
            b_t = max(0.0, training_remaining_budget / max(1, total_samples - t))
            i_lo, i_hi, p_hi, lambda_t = lp_policy_over_estimates(
                ucb, combo_cost, cost_order, b_t)
            # The LP optimum is a mixture of at most two combinations; SAMPLE
            # from it. The randomisation is not incidental -- it is what stops
            # the arms being self-selected, which a deterministic rule cannot
            # avoid. lambda_t is the LP's shadow price on the budget, so the
            # OMD dual ascent below is skipped in this mode.
            j = i_hi if (p_hi > 0.0 and float(rng.random()) < p_hi) else i_lo
            mask = combo_masks[j].copy()
        else:
            diff_mean_sq = pairwise_diff_sq_from_means(acq_centers)  # (nv, nc, nc)
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

        # ── centre / count update (the ONLY place the modes differ) ──
        if center_update == "reward_estimates":
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
                for sub in _submasks_containing(sel_bits, start_bits):
                    sm = np.zeros(nviews, dtype=bool)
                    for i in range(nviews):
                        if sub & (1 << i):
                            sm[i] = True
                    y_sub = predict_mask_multiclass(x[t], est_centers, sm,
                                                    pred_rule=pred_rule)
                    targets.append((sub, float(y_sub == y_true)))
            for bits, r_obs in targets:
                j = bit_index[bits]
                combo_counts[j] += 1.0
                r_hat[j] += (r_obs - r_hat[j]) / combo_counts[j]
        elif center_update == "reward_full":
            # ACQUISITION-ONLY update: acq_centers moves, est_centers (the
            # classifier) does not. Same running-mean rule as "full".
            est_counts[y_true, mask] += 1
            acq_centers[y_true, mask] += (
                (1.0 / est_counts[y_true, mask]) * (x[t][mask] - acq_centers[y_true, mask])
            )
        elif center_update == "reward_bandit":
            # ACQUISITION-ONLY update driven by the 1-bit reward.
            est_counts[y_hat, mask] += 1
            if reward:
                acq_centers[y_hat, mask] += (
                    (1.0 / est_counts[y_hat, mask]) * (x[t][mask] - acq_centers[y_hat, mask])
                )
            else:
                elim = np.zeros(nclasses)
                elim[y_hat] = 1.0
                l_grad = -2 * (x[t][mask][None, :] - acq_centers[:, mask])
                grad = l_grad * (one_vec - (nclasses - 1) * elim)[:, None]
                acq_centers[:, mask] -= lr * grad
        elif center_update == "full":
            # verbatim gmm_multiclass_submodular feedback="full"
            est_counts[y_true, mask] += 1
            est_centers[y_true, mask] += (
                (1.0 / est_counts[y_true, mask]) * (x[t][mask] - est_centers[y_true, mask])
            )
            acq_centers = est_centers  # one shared estimate in this mode
        elif center_update == "bandit":
            # verbatim gmm_multiclass_submodular feedback="bandit"
            est_counts[y_hat, mask] += 1
            if reward:
                est_centers[y_hat, mask] += (
                    (1.0 / est_counts[y_hat, mask]) * (x[t][mask] - est_centers[y_hat, mask])
                )
            else:
                elim = np.zeros(nclasses)
                elim[y_hat] = 1.0
                means_sub = est_centers[:, mask]
                l_grad = -2 * (x[t][mask][None, :] - means_sub)      # (nc, ns)
                grad = l_grad * (one_vec - (nclasses - 1) * elim)[:, None]
                est_centers[:, mask] -= lr * grad
            acq_centers = est_centers  # one shared estimate in this mode
        # center_update == "frozen": nothing to do.

        # DUAL UPDATE -- identical to the EXP4 version.
        if center_update != "reward_estimates":
            # OMD dual ascent. Skipped under reward_estimates, where the LP
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
        'centers_acq': acq_centers,
        'combo_rewards': r_hat,
        'combo_counts': combo_counts,
        'est_counts': est_counts,
        'center_update': center_update,
        'avg_views_acquired': float(np.mean(views_trace)),
        'n_unique_masks': len(seen_masks),
    }