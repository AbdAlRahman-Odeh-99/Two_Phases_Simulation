"""
DIMEOneShotEstimator: a subclass of CMIEstimatorPL2 that changes the
TRAINING OBJECTIVE to match what DIMEOneShot actually does at inference
(query the value network ONCE, rank all paid features, commit to a
budget-constrained subset from that single frozen ranking) -- rather
than training for the original adaptive, re-querying task and only
changing how the trained model gets USED (that's what run_dime's
"Option 1" path does, unchanged, with plain CMIEstimatorPL2).

Only training_step/validation_step are overridden. Everything else
(configure_optimizers, set_feature_costs, set_stopping_criterion,
predict_step, inference) is inherited unchanged from CMIEstimatorPL2.

=== What actually changes vs. the parent class, and why ===

Parent's training_step re-queries the value network at every one of
`max_features` steps, against a mask that keeps growing -- so only the
FIRST query (empty mask) matches what DIMEOneShot's one-shot inference
ever sees. Every later query in training conditions on information
DIMEOneShot's one-shot walk never actually has at decision time.

This class:
  1. Queries the value network ONCE, at the empty mask, per sample.
  2. Ranks ALL paid features once by predicted CMI / cost (with epsilon
     exploration applied to the RANKING itself, not to a per-step pick --
     see _rank_features).
  3. Walks that FIXED ranking to grow the mask, but only to generate a
     TRUE marginal-value training target for each position (via
     before/after predictor loss) -- the value network is never queried
     again after step 1. Its single initial-state output is supervised
     against the true marginal value of EVERY position in the plan, not
     just the next one.
  4. Stops the walk BY BUDGET, per-sample, not by a fixed step count --
     replacing `max_features` (a step-count cap) with `budget` (a cost
     cap), mirroring exactly how max_features disappeared in favor of
     max_budget_ever/budget when your original sequential EDDI was
     converted to eddi_oneshot.py, and how DIMEOneShot itself has no
     max_features, only a budget/max_budget_ever cost ceiling.

     Simplification vs. DIMEOneShot's own inference-time walk: DIMEOneShot
     SKIPS an unaffordable feature and keeps checking cheaper ones further
     down the ranking (a `continue`, not a `break`). This class instead
     TRUNCATES a sample's participation at the first ranking position it
     can't afford (a `break`, per-sample). This is a deliberate
     simplification for clean batched/vectorized training (skip-and-continue
     would need a ragged, per-sample-variable-length walk inside a training
     step, which is awkward to vectorize) -- it does NOT affect
     DIMEOneShot's inference behavior at all, which still uses its own
     skip-and-continue logic, unchanged. Worth knowing about if you're
     comparing training-time vs inference-time budget semantics closely.

Args: same as CMIEstimatorPL2, EXCEPT:
  - `max_features` is REPLACED by `budget` (float, REQUIRED): the cost
    ceiling for the one-shot training walk. Internally, a cheapest-first
    bound (same idea as CAEOneShot's num_select sizing) derives an upper
    limit on walk depth from `budget`, purely so the training loop
    doesn't walk further than any sample could possibly afford -- this
    is an efficiency bound, not a behavioral cap; the REAL stopping rule
    is the per-sample budget check inside the walk.
  - `feature_costs` is REQUIRED (no uniform-cost fallback) -- budget-based
    stopping needs real, meaningful costs to be well-defined.
"""

import torch
import torch.nn.functional as F
from baselines.cmi_estimator_pl2 import CMIEstimatorPL2
from core.utils import get_entropy, ind_to_onehot
from core.budget_state import BudgetState  # noqa: F401 -- type reference only


class DIMEOneShotEstimator(CMIEstimatorPL2):
    def __init__(self,
                 value_network,
                 predictor,
                 mask_layer,
                 lr,
                 budget,
                 eps,
                 loss_fn,
                 val_loss_fn,
                 feature_costs,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None,
                 eps_decay=0.2,
                 eps_steps=1,
                 cmi_scaling='bounded',
                 budget_state=None):
        if feature_costs is None:
            msg = ('DIMEOneShotEstimator requires feature_costs -- budget-based '
                   'training has no meaningful uniform-cost fallback.')
            raise ValueError(msg)

        costs_list = feature_costs.tolist() if torch.is_tensor(feature_costs) else list(feature_costs)
        paid_costs = costs_list[1:]  # exclude index 0 (free modality) -- see class docstring
        sorted_costs = sorted(paid_costs)
        spent, count = 0.0, 0
        for c in sorted_costs:
            if spent + c > budget:
                break
            spent += c
            count += 1
        max_walk_depth = max(1, count)  # efficiency bound only -- see docstring

        super().__init__(
            value_network=value_network, predictor=predictor, mask_layer=mask_layer,
            lr=lr, max_features=max_walk_depth, eps=eps, loss_fn=loss_fn,
            val_loss_fn=val_loss_fn, factor=factor, patience=patience, min_lr=min_lr,
            early_stopping_epochs=early_stopping_epochs, eps_decay=eps_decay,
            eps_steps=eps_steps, feature_costs=feature_costs, cmi_scaling=cmi_scaling,
            budget_state=budget_state)

        self.budget = budget

    def _estimate_cmi_once(self, x, mask):
        '''Single value-network query at the given (typically empty) mask.'''
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        if self.cmi_scaling == 'bounded':
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(x_masked).sigmoid() * entropy
        elif self.cmi_scaling == 'positive':
            pred_cmi = F.softplus(self.value_network(x_masked))
        else:
            pred_cmi = self.value_network(x_masked)
        return pred, pred_cmi

    def _rank_features(self, pred_cmi, mask, training):
        '''
        Rank all features once by predicted CMI/cost. Costs are clamped
        to avoid literal division-by-zero on the free modality's cost=0
        (matching DIMEOneShot.select_features's own
        torch.clamp(feature_costs, min=1e-12)), and already-observed
        modalities (the free one, marked in `mask` before this is ever
        called) are suppressed via -1e6*mask BEFORE ranking -- also
        matching DIMEOneShot's own convention -- so the free modality
        can never be top-ranked or re-selected.

        Epsilon exploration perturbs the RANKING (whole-sample chance of
        using a random ranking instead), not a per-step pick -- since
        there's no per-step decision left to perturb in a one-shot ranking.
        '''
        adjusted_cmi = pred_cmi - 1e6 * mask
        if self.budget_state is not None:
            # Lagrangian budget constraint: price cost via the current
            # OMD lambda instead of cost-normalizing the ranking score.
            # Substituted in place of `adjusted_cmi / feature_costs`.
            scores = self.budget_state.penalize_scores(adjusted_cmi, self.feature_costs)
        else:
            scores = adjusted_cmi / torch.clamp(self.feature_costs, min=1e-12)
        if training and self.eps > 0:
            batch_size = scores.shape[0]
            explore = torch.rand(batch_size, device=scores.device) < self.eps
            if explore.any():
                random_scores = torch.rand_like(scores)
                scores = torch.where(explore.unsqueeze(1), random_scores, scores)
        return torch.argsort(scores, dim=1, descending=True)

    def _one_shot_rollout(self, x, y, training):
        '''
        Single value-network query -> fixed ranking -> budget-limited walk,
        generating a true-marginal-value training target for each walked
        position against the ORIGINAL (never re-queried) pred_cmi.

        IMPORTANT: unlike the parent class (which re-queries pred_cmi fresh
        at every step, so each step's graph is independent), this class
        computes pred_cmi ONCE and every step's loss depends on that SAME
        graph node. Calling manual_backward() more than once against the
        same graph raises "Trying to backward through the graph a second
        time" (confirmed by actually running this) -- so all step losses
        are accumulated into ONE total and backpropagated ONCE, after the
        walk finishes, rather than per-step as the parent class does.
        '''
        device = x.device
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=device)
        mask[:, 0] = 1  # free modality always observed -- matches DIMEOneShot's own convention

        pred0, pred_cmi = self._estimate_cmi_once(x, mask)
        order = self._rank_features(pred_cmi, mask, training)

        loss_prev = self.loss_fn(pred0, y)
        pred_loss_total = loss_prev.mean().detach()
        total_loss = loss_prev.mean() / (self.max_features + 1)
        loss_prev = loss_prev.detach()

        spent = torch.zeros(len(x), device=device)
        active = torch.ones(len(x), dtype=torch.bool, device=device)
        value_network_loss_total = torch.zeros((), device=device)  # must stay a TENSOR even if
        steps_taken = 0                                            # zero steps are taken -- see below
        batch_realized_cost = 0.0  # accumulated for the OMD update at the end of this rollout

        if self.budget_state is not None and training and self.budget_state.is_exhausted:
            # Global pool already empty for this training run -- force the
            # WHOLE batch to fall back to whatever's already in `mask`
            # (the free feature only, at rollout start), matching
            # run_phase1_training's `if remaining_budget <= 0: subset =
            # fallback_subset`. Skip the walk entirely. Gated by
            # `training` for the same reason the per-round clip below is
            # (see its comment) -- validation must never be constrained
            # by a pool that only tracks real training-time spend; it
            # should always reflect what the learned policy would pick.
            active = torch.zeros(len(x), dtype=torch.bool, device=device)

        for k in range(self.max_features):
            if not active.any():
                break

            j = order[:, k]
            cost_j = self.feature_costs[j]
            would_spend = spent + cost_j
            if self.budget_state is not None and training:
                if self.budget_state.remaining_budget is not None:
                    # EXACT per-sample clip: among the samples that ARE
                    # eligible to acquire this round (active), accept
                    # them in order up to however much fits in what's
                    # actually left in the pool -- once accepting a
                    # sample's acquisition would push the running total
                    # past remaining_budget, exclude THAT sample (and
                    # every active sample after it, this round). This
                    # replaces the old "gated only by whole-rollout
                    # exhaustion, checked once before the walk begins"
                    # behavior, which let an entire multi-round rollout's
                    # cost through completely uncapped before the NEXT
                    # rollout would ever see is_exhausted -- see
                    # budget_state.py's generate_budget_constrained_mask
                    # for the same pattern applied to PVAE/the masking
                    # pretrainer.
                    #
                    # Gated by `training` (matching the ORIGINAL code's
                    # scoping for the spend/update_lambda side effects
                    # below) -- validation calls must NEVER touch
                    # remaining_budget or cumulative_spent, since they
                    # don't represent real training-time acquisition. An
                    # earlier version of this fix accidentally dropped
                    # that guard when moving spend() inside this loop,
                    # letting sanity-check and end-of-epoch validation
                    # batches silently drain/contaminate the SAME pool
                    # real training batches were relying on -- exactly
                    # the kind of bug this whole fix was meant to
                    # eliminate.
                    active_costs = torch.where(active, cost_j, torch.zeros_like(cost_j))
                    cumsum = torch.cumsum(active_costs, dim=0)
                    fits = cumsum <= self.budget_state.remaining_budget
                    step_active = active & fits
                else:
                    # Lagrangian-only mode (no total_budget set) -- ranking
                    # is already lambda-penalized (see _rank_features), so
                    # no per-sample cost cutoff needed here.
                    step_active = active
            elif self.budget_state is not None:
                # Validation (training=False): use the SAME acquisition
                # policy (no per-round clip -- validation never touches
                # the pool) so validation metrics reflect what the
                # policy would actually pick, without any budget side
                # effects.
                step_active = active
            else:
                step_active = active & (would_spend <= self.budget)
            if not step_active.any():
                break  # per-sample truncation -- see docstring re: DIMEOneShot's skip-and-continue

            new_mask = mask.clone()
            new_mask[step_active] = torch.max(
                mask[step_active], ind_to_onehot(j[step_active], self.mask_size))

            x_masked_after = self.mask_layer(x, new_mask)
            pred_after = self.predictor(x_masked_after)
            loss_after = self.loss_fn(pred_after, y)

            delta = (loss_prev - loss_after.detach())[step_active]
            picked_cmi = pred_cmi[torch.arange(len(x), device=device), j][step_active]
            value_network_loss = F.mse_loss(picked_cmi, delta)

            step_loss = value_network_loss + loss_after[step_active].mean()
            total_loss = total_loss + step_loss / (self.max_features + 1)

            value_network_loss_total = value_network_loss_total + value_network_loss.detach()
            pred_loss_total = pred_loss_total + loss_after[step_active].mean().detach()

            mask = new_mask
            spent = torch.where(step_active, would_spend, spent)
            loss_prev = torch.where(step_active, loss_after.detach(), loss_prev)
            if self.budget_state is not None and training:
                step_cost = cost_j[step_active].sum().item()
                # Spent IMMEDIATELY (not accumulated until the end of the
                # whole rollout) so the NEXT round's clip check above
                # sees the updated remaining_budget -- this is what makes
                # the cap exact across the whole multi-round walk, not
                # just within one round.
                self.budget_state.spend(step_cost)
                batch_realized_cost += step_cost
            active = step_active
            steps_taken += 1

        if training:
            self.manual_backward(total_loss)

        if self.budget_state is not None and training:
            # spend() already happened per-round, inside the walk above
            # (that's what makes the pool cap exact -- see the loop's
            # comment). Only the OMD lambda update stays here, at the
            # ORIGINAL once-per-rollout cadence, using the accumulated
            # (already-clipped) total realized cost.
            self.budget_state.update_lambda(batch_realized_cost / max(len(x), 1))

        denom = max(steps_taken, 1)
        return value_network_loss_total / denom, pred_loss_total / (steps_taken + 1)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y = batch
        value_network_loss, pred_loss = self._one_shot_rollout(x, y, training=True)
        opt.step()
        out = {'value_network_loss': value_network_loss, 'predictor_loss': pred_loss}
        self._train_step_outputs.append(out)
        return out

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            value_network_loss, pred_loss = self._one_shot_rollout(x, y, training=False)

        # Reuse the parent class's on_validation_epoch_end, which expects
        # (pred_list, y) tuples of length (max_features + 1) for EVERY
        # batch, and zips them positionally across all batches in the
        # epoch. Since this class's walk can stop early per-batch
        # (budget-dependent, unlike the parent's fixed max_features
        # steps), pred_list must be PADDED to a consistent length here --
        # otherwise zip(*pred_list) in the parent's aggregation would
        # silently truncate to the shortest batch's walk length, quietly
        # dropping later positions and misrepresenting "final" stats for
        # every batch that walked further. Padding repeats the last
        # actually-computed prediction (i.e. "we stopped here because the
        # budget ran out, so this is our final answer") for any remaining
        # positions -- semantically correct, not just a shape patch.
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        mask[:, 0] = 1  # free modality always observed -- matches DIMEOneShot's own convention
        pred0, pred_cmi = self._estimate_cmi_once(x, mask)
        order = self._rank_features(pred_cmi, mask, training=False)
        pred_list = [pred0]
        spent = torch.zeros(len(x), device=x.device)
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for k in range(self.max_features):
            if not active.any():
                pred_list.append(pred_list[-1])
                continue
            j = order[:, k]
            cost_j = self.feature_costs[j]
            would_spend = spent + cost_j
            step_active = active & (would_spend <= self.budget)
            if not step_active.any():
                pred_list.append(pred_list[-1])
                active = step_active
                continue
            mask = mask.clone()
            mask[step_active] = torch.max(mask[step_active], ind_to_onehot(j[step_active], self.mask_size))
            x_masked = self.mask_layer(x, mask)
            pred_list.append(self.predictor(x_masked))
            spent = torch.where(step_active, would_spend, spent)
            active = step_active

        out = (pred_list, y)
        self._val_step_outputs.append(out)
        return out