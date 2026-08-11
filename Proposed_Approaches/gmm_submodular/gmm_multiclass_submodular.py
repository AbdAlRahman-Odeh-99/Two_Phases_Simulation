# -*- coding: utf-8 -*-
"""
gmm_multiclass_submodular.py

Training-phase / inference-phase restructuring of the MULTICLASS
submodular-greedy simulation from the Colab notebook
gmm_afa_supervised_full_vs_bandit_feedback.ipynb, aligned with
gmm_2class_submodular_asymmetric.py's phase structure so multiclass runs
slot into the same runner / unified-Excel machinery as the binary
bandit / submodular / two_stage experiments:

  - Training phase (80% of samples): online adaptive acquisition -- the
    notebook's pairwise-Bhattacharyya optimistic scheme (`simulation` /
    `simulation_full_feedback`) with the multiclass `greedy_oracle`,
    under a training budget, with a `feedback` switch:
        feedback="full"   -> simulation_full_feedback's update: y_true is
                             revealed every round; running mean of the
                             TRUE class on acquired views; counts track
                             true-class visitations.
        feedback="bandit" -> simulation's update: only the one-bit reward
                             (y_pred == y_true) is observed; correct ->
                             running-mean update of the predicted (=true)
                             class; incorrect -> complementary-label
                             gradient step with learning rate `lr`;
                             counts track PREDICTED-class visitations.
  - Inference phase (20% of samples): LP-based stochastic inference policy
    via MULTICLASS column generation (core.lp_colgen.
    solve_lp_policy_colgen_multiclass -- same restricted-master + B&B
    pricing as the binary solver, reward swapped to the multiclass
    average pairwise Bhattacharyya proxy), then physically sampled and
    scored against a separate inference budget, exactly mirroring
    gmm_2class_submodular_asymmetric.run_inference_phase.

=== Deliberate changes vs. the notebook (everything else is verbatim) ===
1. BUG FIX in greedy_oracle's giant-item check (the function now lives in
   core/submodular_greedy.py -- see item 8; this note is kept here for
   provenance): the notebook computed
   `final_indices = free_indices + [best_single_item]` with free_indices a
   NUMPY ARRAY, which performs elementwise ADDITION (np.array([0]) + [5]
   -> array([5])), not list concatenation. free_indices is a Python list
   here, restoring the intended "free views + the one giant item"
   semantics (and matching the binary greedy_oracle, where free_indices
   was already a list).
2. Repo phase/budget conventions replace the notebook's single-loop ones:
   training budget pool with per-round OMD target spending_ratio =
   training_budget / n_train (the notebook used the global budget_ratio),
   step_size default 1.0 (binary pipeline convention; the notebook used
   0.2 -- exposed as a parameter if the old dynamics are wanted).
3. NO Hungarian label matching (unlike the binary pipeline): both
   feedback modes anchor class indices to true labels through their
   updates (full: est_means[y_true] updated; bandit: correct-prediction
   updates only fire when y_pred == y_true), so the mapping is the
   identity by construction. label_mapping is still returned (as the
   identity) for interface parity with the binary training-phase dict.
4. Metrics generalized to K classes: F1 is MACRO-averaged; AUROC is
   macro one-vs-rest, computed from softmax(-0.5 * squared distance to
   each class mean on the OBSERVED views) -- the exact class posterior
   under the unit-variance GMM restricted to the acquired views, and
   consistent with pred_linear_cla's pairwise-vote decision (pairwise
   linear votes with shared covariance reduce to nearest-mean, ties
   aside). With nclasses == 2 both metrics fall back to their binary
   forms (positive-class score column / plain binary F1 semantics under
   macro averaging).
5. The first `nclasses` rounds are notebook-style warmup: ALL views are
   acquired (budget permitting, with the notebook's free-views-only
   override otherwise) and the prediction is uniform-random. This warmup
   is part of the multiclass algorithm and is budget-accounted.
6. est_means init follows the repo convention (caller passes
   est_means_init, e.g. nclasses distinct rows of X_train drawn once per
   seed) instead of the notebook's rng.normal weights.
7. @numba.njit instead of bare @jit; Colab scaffolding removed.
8. greedy_oracle, multiclass_reward and bhattacharyya_accuracy_proxy MOVED OUT of
   this file into core/submodular_greedy.py (and re-imported above), so
   two_stage/two_stage_multiclass_greedy.py shares one canonical oracle
   rather than a copy that could drift. The moved greedy_oracle gained a
   `force_free=True` default: the free (zero-cost) views are now ALWAYS
   included, matching the invariant that EXP4's generate_view_combinations
   and the LP's free_mask already enforced. This is a NO-OP at alpha_ucb > 0
   (the optimism bonus is strictly positive, so free views already passed
   the old `margin_gain > 0` test -- verified identical on 3000/3000
   randomized trials), so results produced before this move do NOT need
   re-running.
9. NEW `acquisition` switch on run_training_phase, porting
   two_stage/two_stage_multiclass_greedy.py's
   center_update="reward_estimates" branch into this file so the same
   three acquisition policies can be compared in the ONLINE-LEARNING
   setting (drifting centres) as well as two_stage's frozen-classifier
   setting. See the ACQUISITION MODES section below. acquisition="greedy"
   is the default and is BIT-IDENTICAL to the pre-change code (same
   arithmetic, same rng draw sequence), so existing results stand.

=== ACQUISITION MODES (which subset gets acquired each round) ===
`acquisition` chooses HOW the per-round subset is picked; it is orthogonal
to `feedback`, which chooses how est_means is updated afterwards. All
three modes share the warmup, the budget bookkeeping, the prediction rule
and the metrics.

  "greedy"    (DEFAULT, unchanged) Per-round submodular greedy on the
              OPTIMISTIC pairwise tensor, thresholded by the OMD dual
              variable: core.submodular_greedy.greedy_oracle. The action
              space is implicit -- the subset is re-derived from scratch
              every round and the budget is enforced softly, through the
              dual.
  "lp_full"   ENUMERATE the full action space (all 2^(nviews-1) subsets
              with the free view forced in, via
              core.two_stage_utils.generate_view_combinations), keep a UCB
              reward estimate per arm, and each round solve the per-round
              budgeted LP over those arms EXACTLY
              (core.submodular_greedy.lp_policy_over_estimates). Guarded
              by MAX_REWARD_ESTIMATE_VIEWS: eager enumeration is only
              tractable at small nviews, so this is the FIDELITY CHECK --
              it measures what the chain restriction below costs, on
              identical seeds.
  "lp_chain"  Same UCB + per-round LP, but over the NESTED GREEDY CHAIN
              (core.submodular_greedy.greedy_chain): S_0 = {free views}
              subset S_1 subset ... subset S_p = all views, i.e. nviews+1
              arms instead of 2^(nviews-1). Tractable at any nviews.
  "lp_full_opt"
              ORACLE CEILING for the lp_full family. Same full enumeration,
              but the arm values are not estimated at all: they are the
              EXACT accuracies of the TRUE generative means (caller passes
              true_means; core.optimal_static.synthetic_true_means recovers
              them, so this mode is SYNTHETIC-ONLY). The LP is solved ONCE
              before the round loop and each round is a single draw from
              that frozen distribution -- no UCB, no re-solve, no reward
              update. Subtracting lp_full from it isolates the cost of
              LEARNING the arm values from the cost of the action space and
              of the LP itself. Note it does NOT give the classifier the
              true means: est_means still learns under `feedback` exactly as
              in every other mode, so the oracle sits on the ACQUISITION
              axis alone.

=== PER-ROUND ALLOWANCE (lp_chain / lp_full) ===
The budget the per-round LP is solved against is ADAPTIVE:

    b_allowance = max(0, remaining_budget) / max(1, n_train - t)

recomputed every round from the budget that survived, over the rounds that
remain. It was previously frozen at its first post-warmup value; see the
comment at the assignment for what that cost. "lp_full_opt" is the
deliberate exception -- solving once is its defining property, so it uses
the flat training_budget / n_train and never revisits it.

Two details are specific to THIS file (two_stage has a Stage 1; this
module does not):

  * The action space and the initial reward estimates come from the
    WARMUP rounds. The first `nclasses` rounds already acquire ALL views
    (item 5 above), which is exactly the "fully observed prefix" that
    two_stage's Stage 1 provides, so the chain is built from the centres
    as they stand at the end of warmup and r_hat is seeded by REPLAYING
    those rows (the analogue of stage1_combo_rewards). Warmup rounds that
    got budget-truncated to free views only are excluded from the replay:
    their unacquired views were never paid for, so scoring them would be
    reading data the policy never bought. If NO warmup row was fully
    observed, r_hat starts flat at chance (1/nclasses) with count 1.
  * The arm set is frozen at the end of warmup, but est_means keeps
    moving underneath it (that is the whole point of this module). So
    r_hat is a running mean of a NONSTATIONARY quantity -- an arm's
    accuracy under a classifier that is still learning -- and early
    observations are stale. This is deliberate and is the substantive
    difference from two_stage's version, where the classifier is frozen
    and r_hat estimates a fixed quantity.

`reward_update` chooses what gets scored each round. NOTE that it is
constrained by `feedback`: "subsets" reads y_true, so it cannot be combined
with feedback="bandit" (run_training_phase raises). The remaining three
combinations are all meaningful; see the guard's comment.

  "subsets"  (DEFAULT) COUNTERFACTUAL REPLAY. x[t] was paid for on every
             view of the played set, so every arm CONTAINED IN it can be
             re-scored for free -- many observations per round instead of
             one. Needs y_true, so this is FULL feedback regardless of
             the `feedback` switch (which only governs est_means).
  "selected" BANDIT scope: only the played arm is updated, from the 0/1
             reward already computed. Compatible with feedback="bandit"
             without leaking the label.
"""

import numba
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from core.lp_colgen import (
    multiclass_reward,           # noqa: F401 -- canonical njit'd copy lives in core
    pairwise_diff_sq_from_means,
)

# greedy_oracle and its pairwise-Bhattacharyya risk MOVED to
# core/submodular_greedy.py so that two_stage/two_stage_multiclass_greedy.py
# runs a BIT-IDENTICAL oracle (that module's docstring explains why this
# deliberately breaks the usual duplicate-don't-cross-import convention).
# Re-exported here so existing `from ...gmm_multiclass_submodular import
# greedy_oracle` call sites keep working.
from core.submodular_greedy import (
    # ── shared policy vocabulary; core is the canonical definition so that
    # run_proposed_methods.py's one --acquisition / --reward-update flag
    # means the same thing here and in two_stage_greedy. Re-exported so
    # existing `from ...gmm_multiclass_submodular import ACQUISITION_MODES`
    # call sites keep working.
    ACQUISITION_MODES,             # noqa: F401 -- re-exported
    LP_ACQUISITION_MODES,          # noqa: F401 -- re-exported
    MAX_REWARD_ESTIMATE_VIEWS,     # noqa: F401 -- re-exported
    ORACLE_ACQUISITION_MODES,      # noqa: F401 -- re-exported
    REWARD_UPDATE_SCOPES,          # noqa: F401 -- re-exported
    arm_accuracies_from_means,
    bhattacharyya_accuracy_proxy,      # noqa: F401 -- re-exported
    build_arm_tables,
    greedy_chain,
    greedy_oracle,
    lp_policy_over_estimates,
    linprog_policy_over_estimates,
    mask_to_bits,
    multiclass_reward,               # noqa: F401 -- re-exported
)
from core.two_stage_utils import generate_view_combinations


# ─────────────────────────────────────────────────────────────────────────
# Prediction -- verbatim notebook pairwise-vote rule, plus a per-class
# posterior-score helper for macro-OVR AUROC.
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
    """EAGER per-arm reward estimate from FULLY OBSERVED rows, plus counts.

    Counterpart of two_stage_multiclass_greedy.stage1_combo_rewards: the
    warmup rounds observe ALL views, so the current classifier can be
    REPLAYED over those rows using only the views in each arm. The result
    is that arm's empirical accuracy -- genuine observations on exactly the
    scale the per-round updates use, which is why the counts start at the
    number of replayed rows rather than at a pseudo-count.

    Vectorised NEAREST-CENTRE evaluation. Under a shared unit covariance
    this is the same decision rule as pred_linear_cla's pairwise vote (ties
    aside) -- the same equivalence two_stage_multiclass.py's PRED_RULES
    section documents -- and it is used here only to SEED the estimates;
    every subsequent update goes through pred_linear_cla itself.

    Returns (r_hat, counts), both shape (n_arms,). With no fully observed
    rows, falls back to a flat chance-level prior at count 1.
    """
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
# Training phase: adaptive online training under a training budget, with a
# full/bandit feedback switch and a greedy / LP-over-arms acquisition
# switch.
# ─────────────────────────────────────────────────────────────────────────
def run_training_phase(nviews, nclasses, costs, n_train, training_budget,
                        X_train, Y_train, est_means_init, feedback="full",
                        alpha_ucb=2.0, lr=1e-2, step_size=1.0,
                        lambda_max=10.0, rng=None,
                        acquisition="greedy", reward_update="subsets",
                        force_free=True, true_means=None):
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
    acquisition : {"greedy", "lp_chain", "lp_full", "lp_full_opt"}, default "greedy"
        Per-round subset selection. See the module docstring's ACQUISITION
        MODES section. "greedy" reproduces the previous behaviour exactly.
    true_means : (nclasses, nviews) array, REQUIRED for "lp_full_opt"
        The generative class means. Ignored by every other mode. Supply
        core.optimal_static.synthetic_true_means(...) at the POST-truncation
        view width (pass X_train.shape[1] as n_views_used).
    reward_update : {"subsets", "selected"}, default "subsets"
        Ignored under acquisition="greedy" (there are no arms to score) and
        under "lp_full_opt" (the arm values are exact, so there is nothing
        to score).
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
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")
    # NOTE the guard is scoped to the LEARNING LP modes, not to "anything
    # that is not greedy". lp_full_opt never scores an arm, so reward_update
    # is inert under it and the combination it would otherwise reject
    # (bandit + subsets + lp_full_opt) leaks nothing.
    if (feedback == "bandit" and reward_update == "subsets"
            and acquisition in ("lp_chain", "lp_full")):
        # INCOHERENT COMBINATION, rejected rather than silently run.
        # feedback="bandit" asserts the round observes ONLY the 0/1 reward,
        # but reward_update="subsets" scores every contained arm by calling
        # predict on it and comparing to y_true -- i.e. it reads the label
        # the bandit model says was never revealed. The results would look
        # like bandit feedback while quietly enjoying full feedback on the
        # acquisition policy. The other three cells are fine:
        # full+subsets, full+selected (full-feedback classifier with
        # deliberately bandit-scope arms -- the ablation that separates
        # "the replay helps" from "the LP helps"), and bandit+selected.
        raise ValueError(
            "feedback='bandit' with reward_update='subsets' is incoherent: the "
            "counterfactual replay reads y_true, which bandit feedback does not "
            "reveal. Use reward_update='selected' with feedback='bandit', or "
            "feedback='full' with reward_update='subsets'.")
    # lp_full_opt shares lp_full's action space, so it inherits the same cap.
    if acquisition in ("lp_full", "lp_full_opt") and nviews > MAX_REWARD_ESTIMATE_VIEWS:
        raise ValueError(
            f"acquisition={acquisition!r} enumerates 2^(nviews-1) = 2^{nviews - 1} "
            f"subsets eagerly; nviews={nviews} exceeds "
            f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. Use "
            f"acquisition='lp_chain' (nviews+1 arms), trim with "
            f"max_modalities, or use acquisition='greedy'.")

    is_oracle = acquisition in ORACLE_ACQUISITION_MODES
    if is_oracle:
        # Fail HERE rather than at the LP call: an oracle mode silently
        # falling back to est_means would look like a working run and would
        # quietly stop being an oracle ceiling.
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

    # ── LP-over-arms state (acquisition="lp_chain" / "lp_full" only) ──
    # Everything stays None under acquisition="greedy", which is what makes
    # that branch bit-identical to the pre-change code.
    combo_masks = combo_cost = cost_order = arm_bits = None
    bit_index = r_hat = combo_counts = None
    # Persistent statistics indexed by the subset's bitmask.
    arm_reward_stats = {}
    arm_count_stats = {}
    # b_allowance is now recomputed EVERY round inside the LP branch from the
    # budget that is actually left (see the loop). It is seeded here only so
    # the name exists for the oracle branch and the returned diagnostics.
    b_allowance = spending_ratio
    p_oracle = None
    warmup_rows = []                  # warmup rounds that were FULLY observed
    views_trace = np.zeros(n_train)
    seen_masks = set()

    # ── ORACLE SETUP (acquisition="lp_full_opt" only) ──
    # Everything this mode needs is known before round 0, so all of it
    # happens HERE and the round loop degenerates to a single rng draw.
    #
    # Arm values are the TRUE-means accuracies on the very rows this phase
    # will run over -- an IN-SAMPLE optimum with no estimation error, the
    # same relaxation core.optimal_static documents. There is no UCB bonus
    # because there is nothing to be uncertain about.
    #
    # The allowance is the FLAT training_budget / n_train, and it stays flat:
    # "solve once, outside the loop" and "re-solve against the budget that
    # is left" are mutually exclusive, and this mode is deliberately the
    # former. That makes lp_full_opt the ceiling for NON-ADAPTIVE policies
    # specifically -- it draws from a fixed distribution that never looks at
    # x_t or at how much budget survived. The adaptive lp_chain / lp_full can
    # legitimately land above it; see core.optimal_static's "IS NOT a bound
    # for ADAPTIVE acquisition" note, which applies verbatim here.
    if is_oracle:
        tables = build_arm_tables(generate_view_combinations(nviews), costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        cost_order = tables["cost_order"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]

        r_hat = arm_accuracies_from_means(X_train, Y_train, true_means, combo_masks)
        # Reported as the "count" behind each value. n_train is the honest
        # figure: every arm was scored on every training row.
        combo_counts = np.full(len(arm_bits), float(n_train))

        p_oracle, omd_lambda = linprog_policy_over_estimates(
            r_hat, combo_cost, spending_ratio)

    for t in range(n_train):
        if t < nclasses:
            subset = np.ones(nviews, dtype=bool)
            is_init = True
        elif acquisition == "greedy":
            # optimistic pairwise estimate
            diff_mean_sq = pairwise_diff_sq_from_means(est_means)  # (nv, nc, nc)
            inv_sqrt_cnt = np.sqrt(1.0 / est_counts).T  # (nv, nc)
            bonus_mat = inv_sqrt_cnt[:, :, None] + inv_sqrt_cnt[:, None, :]
            bonus_mat *= np.sqrt(alpha_ucb * np.log(t + 1))
            optimistic_diff_mean_sq = diff_mean_sq + bonus_mat
            subset = greedy_oracle(optimistic_diff_mean_sq, costs, omd_lambda,
                                   remaining_budget, free_indices)
            is_init = False
        elif is_oracle:
            # ── FROZEN ORACLE POLICY ("lp_full_opt") ──
            # The whole round: one draw from the distribution solved above.
            # No arm rebuild, no UCB, no LP, no estimate to update. The only
            # state that still moves is est_means (the classifier keeps
            # learning, exactly as under the other modes) and the budget.
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                subset = combo_masks[int(rng.choice(len(p_oracle), p=p_oracle))].copy()
            is_init = False
        else:
            # ── LP over an ENUMERATED arm set ("lp_chain" / "lp_full") ──
            # Rebuild lp_chain every round. lp_full remains fixed because its
            # action set already contains every allowed subset.
            rebuild_arms = (acquisition == "lp_chain" or combo_masks is None)

            if rebuild_arms:
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

                # Warmup replay provides initialization for subsets that have
                # never appeared in an earlier chain.
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
                ucb = r_hat + np.sqrt(
                    alpha_ucb * np.log(round_idx + 2) / combo_counts)
                #i_lo, i_hi, p_hi, omd_lambda = lp_policy_over_estimates(
                #    ucb, combo_cost, cost_order, b_allowance)
                #j = i_hi if (p_hi > 0.0 and float(rng.random()) < p_hi) else i_lo
                p_lp, omd_lambda = linprog_policy_over_estimates(ucb, combo_cost, b_allowance)
                j = int(rng.choice(len(p_lp), p=p_lp))
                subset = combo_masks[j].copy()
            is_init = False

        # budget check (notebook convention: free-views-only override)
        inst_cost = np.sum(costs[subset])
        if remaining_budget >= inst_cost:
            remaining_budget -= inst_cost
        else:
            subset = free_only_subset.copy()
            inst_cost = np.sum(costs[subset])  # == 0
            remaining_budget -= inst_cost
        total_spent += inst_cost
        views_trace[t] = int(subset.sum())
        seen_masks.add(subset.tobytes())
        if is_init and subset.all():
            # Fully observed warmup round -> replayable. A warmup round that
            # was truncated to free views only is NOT recorded: its other
            # views were never paid for.
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

        # ── per-arm reward-estimate update (LEARNING LP modes only) ──
        # Runs BEFORE the centre update so every arm is scored under the
        # same est_means the played prediction used.
        #
        # `not is_oracle`, NOT just "an arm table exists": lp_full_opt builds
        # combo_masks too, and its r_hat holds the exact true-means values.
        # Averaging observed rewards into those would corrupt the oracle into
        # a slowly-drifting estimate of the LEARNED classifier's accuracy --
        # a silent failure, since the run would still complete and the
        # numbers would still look plausible.
        if combo_masks is not None and not is_oracle:
            played_bits = mask_to_bits(subset)
            if reward_update == "selected":
                # BANDIT scope: one observation, the played arm, from the
                # 0/1 reward already computed. The played subset can fall
                # outside the arm set (the free-views-only override), in
                # which case there is nothing to credit.
                j0 = bit_index.get(played_bits)
                targets = [] if j0 is None else [(j0, float(reward))]
            else:
                # COUNTERFACTUAL REPLAY: x[t] was paid for on every view of
                # the played set, so every arm CONTAINED IN it can be
                # re-scored for free. Vectorised containment test -- O(arms)
                # per round, and no O(arms^2) precomputed submask table.
                targets = []
                for k in np.flatnonzero((arm_bits & played_bits) == arm_bits):
                    m_k = combo_masks[k]
                    y_sub = int(pred_linear_cla(X_train[t, m_k],
                                                est_means[:, m_k]))
                    targets.append((int(k), float(y_sub == y_true)))
            
            for j0, r_obs in targets:
                combo_counts[j0] += 1.0
                r_hat[j0] += (r_obs - r_hat[j0]) / combo_counts[j0]
                # Preserve statistics under the subset identity.
                key = int(arm_bits[j0])
                arm_reward_stats[key] = float(r_hat[j0])
                arm_count_stats[key] = float(combo_counts[j0])

        # ── update (the ONLY place the two feedback modes differ) ──
        if feedback == "full":
            # y_true revealed every round: running mean of the TRUE class
            est_counts[y_true, subset] += 1
            est_means[y_true, subset] += (
                (1.0 / est_counts[y_true, subset]) * (x_obs - est_means[y_true, subset])
            )
        else:  # bandit
            est_counts[y_pred, subset] += 1
            if reward:
                # correct prediction: y_pred == y_true, running-mean update
                est_means[y_pred, subset] += (
                    (1.0 / est_counts[y_pred, subset]) * (x_obs - est_means[y_pred, subset])
                )
            else:
                # incorrect prediction (complementary-label update): the
                # model only knows its prediction was wrong.
                eliminated = y_pred
                elim = np.zeros(nclasses)
                elim[eliminated] = 1.0
                l_grad = -2 * (x_obs[None, :] - means_sub)  # (nc, ns)
                grad = l_grad * (one_vec - (nclasses - 1) * elim)[:, None]
                est_means[:, subset] -= lr * grad

        # OMD dual update (repo convention: per-round target spending_ratio).
        # Skipped under the LP modes, where the per-round LP already enforces
        # the budget exactly and omd_lambda is its shadow price -- a learned
        # dual on top of that would double-count the constraint.
        if acquisition == "greedy":
            raw_lambda = omd_lambda + step_size * (inst_cost - spending_ratio)
            omd_lambda = max(0, min(lambda_max, raw_lambda))

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
        "reward_update": ("" if acquisition in ("greedy",) + ORACLE_ACQUISITION_MODES
                          else reward_update),
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
    }


# ─────────────────────────────────────────────────────────────────────────
# Inference phase: LP-based inference policy + physical sampling
# (masks/probs come from core.lp_colgen.solve_lp_policy_colgen_multiclass)
# -- structural mirror of the binary run_inference_phase.
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