"""
Shared budget-enforcement mechanism, used to make EDDI/CAE/DIME's TRAINING
conditions match the GMM-bandit method's Phase 1 (run_phase1_training in
gmm_2class_bandit_asymmetric.py) as closely as possible, and to make
INFERENCE-time budget accounting match Phase 2's global-depletion/fallback
behavior -- WITHOUT importing Phase 2's LP-solve-and-sample machinery,
which stays specific to the GMM-bandit method.

Two independent things are being ported here, deliberately kept separate:

  1. TRAINING: dual-ascent (a first-order / Euclidean-mirror-map special
     case of "OMD") update of a single scalar price `omd_lambda`, exactly
     matching run_phase1_training's own update rule:

         raw_lambda = omd_lambda + step_size * (instant_cost - spending_ratio)
         omd_lambda = clip(raw_lambda, 0, lambda_max)

     `omd_lambda` prices cost into whatever score each baseline already
     uses to decide what to acquire (DIME's pred_cmi, EDDI's per-feature
     criterion, CAE's selection logits) -- see `penalize_scores` below.
     This is a LAGRANGIAN RELAXATION of the budget constraint: lambda
     reacts to realized spend, but there is no hard per-round cutoff
     enforced through lambda alone.

  2. GLOBAL DEPLETION + FALLBACK: a literal remaining_budget counter that
     depletes as cost is realized (training OR inference), with a forced
     fallback to the free-only feature once it hits zero -- matching both
     run_phase1_training's `if remaining_budget <= 0: subset = fallback`
     and run_phase2_inference's `if remaining_budget - cost < 0: subset =
     fallback_subset`. This is what actually turns a PER-SAMPLE budget
     into a GLOBAL one: the pool is shared and stateful across an ordered
     stream of samples/batches, not reset for each new sample.

Calibration (spending_ratio, lambda_max, step_size) is FIXED for now, per
your instruction -- no auto-tuning. Pass different BudgetState instances
(with dataset-appropriate spending_ratio) per dataset/method rather than
sharing one instance's constants across differently-scaled cost models.
"""

from __future__ import annotations

import torch


class BudgetState:
    """
    Shared training-time (Lagrangian) and inference-time (depleting-pool)
    budget-enforcement state.

    Args:
      spending_ratio: target mean cost per round/sample (the analogue of
        run_phase1_training's `spending_ratio = training_budget / n_train`
        or Phase 2's `inference_budget / n_inference`). Used ONLY to drive
        the OMD dual-ascent update -- it does not by itself cap anything.
      total_budget: if given, initializes `remaining_budget` to this value
        and enables global depletion + fallback (item 2 above). If None,
        only the OMD lambda-pricing mechanism (item 1) is active and
        nothing is ever hard-capped -- useful if you want the Lagrangian
        relaxation without also imposing a literal depleting pool (e.g.
        while still deciding on (b) vs (a) train-time semantics).
      lambda_max: clip ceiling for omd_lambda (matches run_phase1_training's
        `lambda_max = 10`).
      step_size: dual-ascent step size (matches `step_size = 1.0`).
    """

    def __init__(self, spending_ratio, total_budget=None, lambda_max=10.0, step_size=1.0):
        self.spending_ratio = float(spending_ratio)
        self.lambda_max = float(lambda_max)
        self.step_size = float(step_size)
        self.omd_lambda = 0.0

        self.total_budget = None if total_budget is None else float(total_budget)
        self.remaining_budget = self.total_budget

        # Cumulative spend across the ENTIRE lifetime of this object, never
        # reset by reset_pool() -- unlike remaining_budget (which resets
        # per-epoch when reset_pool() is called), this is what you want
        # for "how much was actually spent in total during training",
        # since per-epoch pool resets would otherwise make remaining_budget
        # only reflect the LAST epoch's spend, silently understating the
        # true multi-epoch total.
        self.cumulative_spent = 0.0

    # ------------------------------------------------------------------
    # Item 1: OMD / dual-ascent update on the price lambda
    # ------------------------------------------------------------------
    def update_lambda(self, realized_cost):
        """
        One dual-ascent step, given the realized mean cost of the round/
        batch just completed. Identical update rule to
        run_phase1_training's per-round lambda update, just called once
        per training round (which may be a batch, not a single sample --
        see per-file docstrings for what counts as a "round" there).
        """
        raw_lambda = self.omd_lambda + self.step_size * (float(realized_cost) - self.spending_ratio)
        self.omd_lambda = max(0.0, min(self.lambda_max, raw_lambda))

    def penalize_scores(self, scores, costs):
        """
        Subtract the current lambda-price from a batch of per-feature (or
        per-subset) scores, e.g. DIME's pred_cmi or EDDI's per-feature
        criterion, BEFORE ranking/argmax. Shapes broadcast the same way
        `scores / feature_costs` already does elsewhere in this codebase
        -- this is a straight substitution of that division for a
        subtraction-by-lambda-times-cost, i.e. the Lagrangian penalty
        instead of a hard per-sample cost-normalized ranking.
        """
        return scores - self.omd_lambda * costs

    # ------------------------------------------------------------------
    # Item 2: global depletion + forced fallback (training or inference)
    # ------------------------------------------------------------------
    @property
    def is_exhausted(self):
        """True once the global pool has hit zero. Always False if this
        BudgetState was constructed without a total_budget (pure
        Lagrangian mode, item 1 only)."""
        return self.remaining_budget is not None and self.remaining_budget <= 0

    def spend(self, realized_cost):
        """Deplete the global pool by a realized cost (scalar, e.g. a
        batch's mean cost during training, or a single sample's realized
        cost during inference). No-op on remaining_budget if total_budget
        was never set, but cumulative_spent is always tracked regardless."""
        realized_cost = float(realized_cost)
        self.cumulative_spent += realized_cost
        if self.remaining_budget is not None:
            self.remaining_budget = max(0.0, self.remaining_budget - realized_cost)

    def reset_pool(self, total_budget=None):
        """Reset remaining_budget to a fresh pool (e.g. at the start of a
        new epoch, if you want depletion to be tracked per-epoch rather
        than across the whole training run -- either is defensible, just
        be consistent and say which one you used)."""
        if total_budget is not None:
            self.total_budget = float(total_budget)
        self.remaining_budget = self.total_budget

    def summary(self):
        """
        Snapshot of this state's current lambda/spend, for logging or
        saving to results -- e.g. into a CSV row, so you can check
        post-hoc whether lambda actually moved/settled and how much was
        actually spent, rather than assuming the mechanism behaved as
        intended.

        cumulative_spent is the TRUE total realized cost across this
        object's entire lifetime (survives per-epoch pool resets) -- use
        this as "actual spending", not spent_total.
        spent_total (total_budget - remaining_budget) reflects only the
        CURRENT pool since its last reset_pool() call -- with per-epoch
        resets, this is just the LAST epoch's spend, not the multi-epoch
        total. Kept for reference but cumulative_spent is the one to
        report as "how much was actually spent".
        spent_total/total_budget/remaining_budget are None if this
        BudgetState was never given a total_budget (pure Lagrangian mode).
        """
        spent_total = None
        if self.total_budget is not None:
            spent_total = self.total_budget - self.remaining_budget
        return {
            "lambda_final": self.omd_lambda,
            "cumulative_spent": self.cumulative_spent,
            "total_budget": self.total_budget,
            "remaining_budget": self.remaining_budget,
            "spent_total": spent_total,
        }


# ----------------------------------------------------------------------
# Cost-aware Bernoulli mask generator -- drop-in cost-aware replacement
# for utils.generate_uniform_mask, used by PVAE.fit and
# MaskingPretrainerPL2.training_step.
# ----------------------------------------------------------------------
def generate_budget_constrained_mask(batch_size, costs, state: BudgetState, device=None):
    """
    Cost-aware, budget-depleting replacement for
    utils.generate_uniform_mask(batch_size, num_features).

    Mechanism (mirrors run_phase1_training's get_subset + remaining_budget
    fallback, adapted from "enumerate all subsets" -- fine for the GMM
    script's small nviews -- to independent per-feature Bernoulli
    inclusion, since these baselines' feature counts are too large to
    enumerate 2**n subsets):

      1. Each PAID feature j is included independently with probability
         sigmoid(-omd_lambda * cost_j) -- higher lambda or higher cost
         both push inclusion probability down, mirroring how
         `get_subset`'s reward penalty `lambda_dual * (tmp_cost -
         spending_ratio)` disfavors expensive subsets as lambda rises.
      2. The free feature (index 0, cost 0) is always included, matching
         every other file's free-modality convention.
      3. GLOBAL fallback: if state has a total_budget and it's already
         exhausted, every sample in this batch is forced to the
         free-only mask -- same semantics as run_phase1_training's
         `if remaining_budget <= 0: subset = fallback_subset`.
      4. EXACT per-sample clip WITHIN the batch: even when not YET
         exhausted, the batch's drawn masks are accepted in order
         (cumulative sum of realized cost across the batch) only up to
         however much actually fits in state.remaining_budget -- the
         first sample whose inclusion would push the running total past
         what's left, and every sample after it in this batch, gets
         forced to the free-only fallback instead. This is what makes
         the cap EXACT (state.spend() below can mathematically never
         push remaining_budget negative) rather than the old behavior of
         accepting a whole batch's realized cost unconditionally and
         only blocking the NEXT batch once already over -- which is what
         let cumulative_spent exceed total_budget by however much a
         single un-clipped batch happened to cost.
      5. The batch's mean (post-clip) realized cost is used to (a)
         deplete state.remaining_budget (item 2) and (b) drive the OMD
         lambda update (item 1), once per call -- i.e. once per training
         round, where a "round" here is a minibatch (see per-file
         docstrings on why batch-granularity rounds were used instead of
         literal single-sample rounds).

    Returns: mask, shape (batch_size, num_features), same dtype/device
    contract as generate_uniform_mask.
    """
    if not torch.is_tensor(costs):
        costs = torch.tensor(costs, dtype=torch.float32)
    if device is not None:
        costs = costs.to(device)
    num_features = costs.shape[0]

    if state.is_exhausted:
        # Global pool already empty -- forced fallback for the WHOLE batch,
        # exactly like run_phase1_training's fallback_subset branch.
        mask = torch.zeros(batch_size, num_features, device=costs.device)
        mask[:, 0] = 1.0
        state.update_lambda(0.0)  # realized cost is 0 under forced fallback
        return mask

    # Inclusion probability exp(-lambda*cost): 1.0 at lambda=0 (everything
    # included, matching an unconstrained pass), priced down smoothly as
    # lambda rises. NOTE: an earlier draft used sigmoid(-lambda*cost),
    # which caps inclusion probability at 0.5 since lambda >= 0 -- making
    # any spending target above 0.5 * total_paid_cost unreachable (lambda
    # pins at 0, constraint silently never binds). exp has no such ceiling.
    probs = torch.exp(-state.omd_lambda * costs)
    probs = probs.clone()
    probs[0] = 1.0  # free feature always included

    mask = torch.bernoulli(probs.unsqueeze(0).expand(batch_size, -1).clone())

    realized_cost_per_sample = (mask * costs.unsqueeze(0)).sum(dim=1)

    if state.remaining_budget is not None:
        # Exact clip (item 4 above): accept samples in order up to
        # however much fits in what's actually left.
        cumsum = torch.cumsum(realized_cost_per_sample, dim=0)
        fits = cumsum <= state.remaining_budget
        mask = torch.where(fits.unsqueeze(1), mask, torch.zeros_like(mask))
        mask[~fits, 0] = 1.0  # forced-fallback samples still get the free feature
        realized_cost_per_sample = (mask * costs.unsqueeze(0)).sum(dim=1)  # recompute post-clip

    mean_realized_cost = realized_cost_per_sample.mean().item()
    total_realized_cost = realized_cost_per_sample.sum().item()

    # UNITS: the pool (total_budget) is sized in TOTAL units (per-sample
    # budget * n_samples), so depletion must use the batch TOTAL. The
    # lambda update compares against spending_ratio, a PER-SAMPLE target,
    # so it uses the batch MEAN. Mixing these up makes the pool deplete
    # ~batch_size x too slowly (a real bug caught in review).
    state.spend(total_realized_cost)
    state.update_lambda(mean_realized_cost)

    return mask


# ----------------------------------------------------------------------
# Inference-side global depletion + fallback wrapper.
#
# Each baseline keeps its OWN selection mechanism (DIME's ranking, EDDI's
# subset scoring, CAE's learned mask) -- this wrapper only decides whether
# a sample's ALREADY-CHOSEN subset is affordable against the SHARED,
# depleting pool, exactly mirroring run_phase2_inference's:
#
#     if remaining_budget - cost < 0:
#         subset = fallback_subset
#         cost = fallback_cost
#     remaining_budget -= cost
#
# -- i.e. forced fallback to the free-only subset, NOT skip-and-try-a-
# cheaper-option (that distinction was flagged earlier as a deliberate
# design choice in the GMM script; this wrapper matches it exactly so
# baseline inference-time accounting is the same mechanism).
# ----------------------------------------------------------------------
class MaskPool:
    """
    Single-pass-acquire, then-permute mask pool.

    Epoch 0 spends `state`'s budget EXACTLY ONCE: one call to
    generate_budget_constrained_mask per batch, in loader order,
    accumulated into a fixed pool of shape (num_samples, num_features)
    covering every sample seen that epoch. From epoch 1 onward, NO
    further budget is spent and generate_budget_constrained_mask is
    never called again -- the pool is simply reshuffled (a permutation,
    not a resample) and re-paired with whatever samples the loader
    yields that epoch.

    Why this is valid for PVAE / MaskingPretrainerPL2 specifically:
    neither's mask generation depends on the sample's content (x is
    never read by generate_budget_constrained_mask), so a mask isn't
    "earned by" a particular sample -- it's just one draw from a
    budget-respecting distribution over patterns. Re-pairing pool rows
    with samples each epoch is therefore a free source of mask
    diversity across epochs (which both PVAE and the masking predictor
    need, since they're trained to generalize over many masks, not to
    one fixed acquired subset per sample) while keeping the AGGREGATE
    cost across the entire training run capped at whatever
    state.total_budget was set to -- it can never be spent twice.

    This is deliberately NOT the same mechanism you'd want for a model
    that trains on each sample's own real, content-dependent acquired
    subset (e.g. if you ever train something directly downstream of
    EDDI/DIME/CAE's per-sample selection) -- there, you'd freeze each
    sample to ITS OWN mask, not permute a shared pool across samples.
    Neither EDDI, DIMEOneShot, nor CAE trains anything themselves in
    this codebase (EDDI.fit() raises NotImplementedError), so that case
    doesn't currently arise -- but don't reach for MaskPool if it ever
    does.

    Requires the total number of samples seen per epoch to be constant
    (true for a fixed dataset/DataLoader with consistent drop_last
    behavior, regardless of whether it shuffles -- shuffling is fine
    here specifically because pairing is content-independent).

    Usage (manual loop, e.g. PVAE.fit):
        pool = MaskPool(feature_costs, budget_state)
        for epoch in range(nepochs):
            pool.start_epoch(epoch)
            for x, _ in loader:
                mask = pool.get_mask(len(x), device=x.device)
                ...
            pool.end_epoch(epoch)

    Usage (Lightning hooks, e.g. MaskingPretrainerPL2):
        on_train_epoch_start: pool.start_epoch(self.current_epoch)
        training_step:        pool.get_mask(len(x), device=x.device)
        on_train_epoch_end:   pool.end_epoch(self.current_epoch)
    """

    def __init__(self, feature_costs, state: BudgetState):
        self.feature_costs = feature_costs
        self.state = state
        self._pool = None       # (N, num_features) -- fixed after epoch 0
        self._permuted = None   # this epoch's shuffled view of _pool
        self._chunks = None     # accumulator, epoch 0 only
        self._ptr = 0

    def start_epoch(self, epoch):
        if epoch == 0:
            self._chunks = []
        else:
            if self._pool is None:
                raise RuntimeError(
                    'MaskPool.start_epoch(epoch > 0) called before epoch 0 '
                    'finished -- the pool was never built. Epoch 0 must run '
                    '(start_epoch(0) ... end_epoch(0)) before any later epoch.')
            perm = torch.randperm(self._pool.shape[0])
            self._permuted = self._pool[perm]
        self._ptr = 0

    def get_mask(self, batch_size, device=None):
        if self._chunks is not None:
            # Still epoch 0: real acquisition -- spends state's budget
            # and updates omd_lambda, exactly as before.
            mask = generate_budget_constrained_mask(
                batch_size, self.feature_costs, self.state, device=device)
            self._chunks.append(mask.detach().cpu())
            return mask
        # Epoch > 0: replay from this epoch's permuted pool. No spend,
        # no lambda update -- budget_state is not touched at all.
        mask = self._permuted[self._ptr:self._ptr + batch_size]
        self._ptr += batch_size
        if device is not None:
            mask = mask.to(device)
        return mask

    def end_epoch(self, epoch):
        if epoch == 0:
            if not self._chunks:
                raise RuntimeError('MaskPool: epoch 0 ended with no masks accumulated.')
            self._pool = torch.cat(self._chunks, dim=0)
            self._chunks = None


def apply_global_budget_fallback(cost, state: BudgetState, fallback_cost=0.0):
    """
    Args:
      cost: realized cost of the subset a baseline ALREADY selected for
        one sample.
      state: BudgetState with total_budget set (i.e. remaining_budget is
        being tracked). If state.total_budget is None, this is a no-op
        that always accepts the given cost (no global pool to check).
      fallback_cost: cost of the free-only fallback (0.0 in every
        convention used across these files).

    Returns: (accepted_cost, used_fallback) -- accepted_cost is what
    actually gets charged to the pool (either `cost` or `fallback_cost`);
    used_fallback tells the caller whether to swap its chosen mask/subset
    for the free-only one before returning predictions.
    """
    if state.remaining_budget is None:
        return cost, False

    if state.remaining_budget - cost < 0:
        # BUG FIX: this used to mutate state.remaining_budget directly,
        # which skipped state.cumulative_spent entirely -- meaning
        # inference-side "actual spending" silently read 0 regardless of
        # what was really spent. Routing through state.spend() keeps
        # both counters consistent, the same as every training-side call.
        state.spend(fallback_cost)
        return fallback_cost, True

    state.spend(cost)
    return cost, False