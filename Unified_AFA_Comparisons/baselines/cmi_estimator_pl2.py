"""
Lightning 2.x-compatible port of cmi_estimator.py.

Same two mechanical changes as masking_pretrainer_pl2.py:
  1. training_epoch_end/validation_epoch_end -> on_train_epoch_end/
     on_validation_epoch_end with manually-accumulated outputs.
  2. ReduceLROnPlateau's verbose=True removed.

training_step/validation_step/predict_step logic is otherwise UNCHANGED
from your original cmi_estimator.py -- this is still the adaptive,
re-querying CMIEstimator (Option 1's model). DIMEOneShotEstimator (in
dime_oneshot_estimator.py) subclasses THIS class and overrides
training_step/validation_step for Option 2 -- it does not touch
predict_step, configure_optimizers, or set_feature_costs/
set_stopping_criterion, all inherited unchanged.

One additional note vs. the original: self.logger.experiment.add_scalar(...)
calls are kept as-is. If you run with logger=False, these will raise
(logger.experiment is None) -- don't disable the logger when training
either this class or DIMEOneShotEstimator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from core.utils import get_entropy, get_confidence, ind_to_onehot
from core.budget_state import BudgetState  # noqa: F401 -- imported for type reference in docstrings/callers


class CMIEstimatorPL2(pl.LightningModule):
    '''
    Greedy CMI estimation module. (Lightning 2.x-compatible port of
    CMIEstimator -- see module docstring. Logic is otherwise identical
    to your original.)

    Args: identical to the original CMIEstimator.
    '''

    def __init__(self,
                 value_network,
                 predictor,
                 mask_layer,
                 lr,
                 max_features,
                 eps,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None,
                 eps_decay=0.2,
                 eps_steps=1,
                 feature_costs=None,
                 cmi_scaling='bounded',
                 budget_state=None):
        super().__init__()

        self.value_network = value_network
        self.predictor = predictor
        self.mask_layer = mask_layer

        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs
        self.factor = factor

        self.max_features = max_features
        self.mask_size = self.mask_layer.mask_size
        self.set_feature_costs(feature_costs)

        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_steps = eps_steps
        if cmi_scaling not in ('none', 'positive', 'bounded'):
            raise ValueError('cmi_scaling must be one of "none", "positive", or "bounded"')
        self.cmi_scaling = cmi_scaling

        # Optional Lagrangian budget constraint (see training_step). If
        # None (default), selection is UNCHANGED: argmax(pred_cmi /
        # feature_costs), no lambda pricing, no global depletion.
        self.budget_state = budget_state

        self.automatic_optimization = False

        self._train_step_outputs = []
        self._val_step_outputs = []

    def set_feature_costs(self, feature_costs=None):
        '''Set feature cost values. Default is uniform cost.'''
        if feature_costs is None:
            feature_costs = torch.ones(self.mask_size)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs)
        self.register_buffer('feature_costs', feature_costs)

    def set_stopping_criterion(self, budget=None, lam=None, confidence=None):
        '''Set parameters for stopping criterion.'''
        if sum([budget is None, lam is None, confidence is None]) != 2:
            raise ValueError('Must specify exactly one of budget, lam, and confidence')
        if budget is not None:
            self.budget = budget
            self.mode = 'budget'
        elif lam is not None:
            self.lam = lam
            self.mode = 'penalized'
        elif confidence is not None:
            self.confidence = confidence
            self.mode = 'confidence'

    def on_fit_start(self):
        self.num_epsilon_steps = 0
        self.num_bad_epochs = 0

    def on_train_start(self):
        # Amortize the SAME total_budget across the WHOLE training run
        # (all planned epochs), not one epoch. training_step calls
        # spend()/update_lambda() every batch, every epoch -- so leaving
        # spending_ratio at a per-EPOCH target (as the caller originally
        # set it, e.g. train_budget / n_train) makes the SAME total
        # budget get "spent" roughly once per epoch, ballooning
        # cumulative_spent to roughly (epochs_run x train_budget) by the
        # end of training -- this was a real bug (this file was never
        # updated when the equivalent fix was made in pvae.py /
        # masking_pretrainer_pl2.py / cae_oneshot.py). Rescaling by the
        # planned total epoch count (self.trainer.max_epochs -- an upper
        # bound, since early stopping can only reduce the actual epochs
        # run) makes the dual-ascent price target the correct RATE for a
        # run-wide cap instead of a per-epoch one. Restored in
        # on_train_end, in case this BudgetState instance is reused
        # elsewhere afterward.
        #
        # This hook (not on_fit_start) is used specifically because it's
        # the first point at which self.trainer.train_dataloader is
        # guaranteed to be attached, which is needed to get the exact
        # per-epoch sample count.
        if self.budget_state is not None:
            n_train = len(self.trainer.train_dataloader.dataset)
            total_planned_epochs = max(1, self.trainer.max_epochs or 1)
            self._original_spending_ratio = self.budget_state.spending_ratio
            self.budget_state.spending_ratio = self.budget_state.total_budget / (
                n_train * total_planned_epochs)
            # Pool spans the ENTIRE run -- reset ONCE here, not per
            # epoch (see the removed reset_pool() call that used to be
            # in on_train_epoch_start).
            self.budget_state.reset_pool()

    def on_train_end(self):
        if self.budget_state is not None and hasattr(self, '_original_spending_ratio'):
            self.budget_state.spending_ratio = self._original_spending_ratio

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0

        x_masked = self.mask_layer(x, mask)
        pred_without_next_feature = self.predictor(x_masked)
        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)
        pred_loss = loss_without_next_feature.mean()
        pred_loss_total += pred_loss.detach()
        self.manual_backward(pred_loss / (self.max_features + 1))
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        batch_realized_cost = 0.0  # accumulated per-round realized cost, for the OMD update below

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
                pred_cmi = self.value_network(x_masked).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked))
            else:
                pred_cmi = self.value_network(x_masked)

            if self.budget_state is not None:
                # Lagrangian budget constraint: price cost via the
                # current OMD lambda instead of cost-normalizing the
                # score. Substituted in place of `pred_cmi / feature_costs`.
                selection_score = self.budget_state.penalize_scores(pred_cmi, self.feature_costs)
            else:
                selection_score = pred_cmi / self.feature_costs

            best = torch.argmax(selection_score, dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random

            if self.budget_state is not None:
                # EXACT per-sample clip: accept this round's actions in
                # order, up to however much fits in what's actually left
                # in the pool -- once accepting a sample's action would
                # push the running total past remaining_budget, force
                # THAT sample (and every sample after it, this round) to
                # the free feature (index 0) instead of letting the
                # whole batch's round-cost through uncapped. This
                # replaces the old whole-batch-or-nothing is_exhausted
                # check below it (still correct as a special case: when
                # remaining_budget is already 0, cumsum <= 0 is False for
                # every sample with a nonzero-cost action, so everyone
                # gets forced to free anyway) -- see budget_state.py's
                # generate_budget_constrained_mask for the same pattern.
                step_costs_per_sample = self.feature_costs[actions]
                if self.budget_state.remaining_budget is not None:
                    cumsum = torch.cumsum(step_costs_per_sample, dim=0)
                    fits = cumsum <= self.budget_state.remaining_budget
                    actions = torch.where(fits, actions, torch.zeros_like(actions))
                step_cost = self.feature_costs[actions].sum().item()  # recompute post-clip
                # Spent IMMEDIATELY (not accumulated until end of batch)
                # so the NEXT round's clip check sees the updated
                # remaining_budget -- this is what makes the cap exact
                # across the whole multi-round walk, not just per round.
                self.budget_state.spend(step_cost)
                batch_realized_cost += step_cost

            mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred_with_next_feature = self.predictor(x_masked)
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            delta = loss_without_next_feature - loss_with_next_feature.detach()
            value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x)), actions], delta)

            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
            self.manual_backward(total_loss / (self.max_features + 1))

            value_network_loss_total += torch.mean(value_network_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

        opt.step()

        if self.budget_state is not None:
            # spend() already happened per-round, inside the walk above
            # (that's what makes the pool cap exact -- see the loop's
            # comment). The OMD lambda update stays at the ORIGINAL
            # once-per-batch cadence here (not once per round) to
            # preserve step_size's calibrated meaning -- lambda reacts to
            # this whole batch's (already-clipped) realized cost, same
            # update frequency as before this fix, just fed the correct
            # (now never-overshooting) cost.
            self.budget_state.update_lambda(batch_realized_cost / max(len(x), 1))

        out = {
            'value_network_loss': (value_network_loss_total / self.max_features).detach(),
            'predictor_loss': (pred_loss_total / (self.max_features + 1)).detach()}
        self._train_step_outputs.append(out)
        return out

    def on_train_epoch_start(self):
        # Pool reset used to happen here, every epoch -- see on_train_start
        # for why that was a bug (silently uncapped multi-epoch overspend)
        # and where the fix now lives (reset ONCE, at the start of the
        # whole run, with spending_ratio amortized across all planned
        # epochs instead of treated as a per-epoch target).
        pass

    def on_train_epoch_end(self):
        outputs = self._train_step_outputs
        pred_loss = torch.stack([out['predictor_loss'] for out in outputs]).mean()
        value_network_loss = torch.stack([out['value_network_loss'] for out in outputs]).mean()

        self.log('Loss Train/Mean', pred_loss, prog_bar=True, logger=False)
        self.log('Value Loss Train/Mean', value_network_loss, prog_bar=True, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss Train/Mean', pred_loss, self.current_epoch)
            self.logger.experiment.add_scalar('Value Loss Train/Mean', value_network_loss, self.current_epoch)

        outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        pred_list = [pred]

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred).unsqueeze(1)
                pred_cmi = self.value_network(x_masked).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked))
            else:
                pred_cmi = self.value_network(x_masked)

            pred_cmi -= 1e6 * mask
            best_feature_index = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked)
            pred_list.append(pred)

        out = (pred_list, y)
        self._val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        pred_list, y_list = zip(*self._val_step_outputs)
        y = torch.cat(y_list)
        preds_cat = [torch.cat(preds) for preds in zip(*pred_list)]
        pred_loss_ = [self.loss_fn(preds, y).mean() for preds in preds_cat]
        val_loss_ = [self.val_loss_fn(preds, y) for preds in preds_cat]
        pred_loss_mean = torch.stack(pred_loss_).mean()
        val_loss_mean = torch.stack(val_loss_).mean()
        pred_loss_final = pred_loss_[-1]
        val_loss_final = val_loss_[-1]

        self.log('Loss Val/Mean', pred_loss_mean, prog_bar=True, logger=False)
        self.log('Perf Val/Mean', val_loss_mean, prog_bar=True, logger=False)
        self.log('Loss Val/Final', pred_loss_final, prog_bar=True, logger=False)
        self.log('Perf Val/Final', val_loss_final, prog_bar=True, logger=False)
        self.log('Eps Value', self.eps, prog_bar=False, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss Val/Mean', pred_loss_mean, self.current_epoch)
            self.logger.experiment.add_scalar('Perf Val/Mean', val_loss_mean, self.current_epoch)
            self.logger.experiment.add_scalar('Loss Val/Final', pred_loss_final, self.current_epoch)
            self.logger.experiment.add_scalar('Perf Val/Final', val_loss_final, self.current_epoch)
            self.logger.experiment.add_scalar('Eps Value', self.eps, self.current_epoch)

        sch = self.lr_schedulers()
        sch.step(pred_loss_mean)

        if pred_loss_mean == sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            self.eps *= self.eps_decay
            self.num_bad_epochs = 0
            self.num_epsilon_steps += 1
            print(f'Decaying eps to {self.eps:.5f}, step = {self.num_epsilon_steps}')

            if self.num_epsilon_steps >= self.eps_steps:
                self.trainer.should_stop = True

            for g in self.optimizers().param_groups:
                g['lr'] = self.lr

        self._val_step_outputs.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }

    def on_predict_start(self):
        if not hasattr(self, 'mode'):
            print('Must specify stopping criterion. Recommended usage is via `inference` function')

    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        accept_sample = torch.ones(len(x), dtype=bool, device=x.device)

        for step in range(self.mask_size):
            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred).unsqueeze(1)
                pred_cmi = self.value_network(x_masked).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked))
            else:
                pred_cmi = self.value_network(x_masked)

            check_pos_pred_cmi = pred_cmi.max(dim=1).values >= 0

            pred_cmi -= 1e6 * mask
            best_feature_index = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            selection = ind_to_onehot(best_feature_index, self.mask_size)

            if self.mode == 'penalized':
                accept_sample = torch.max(pred_cmi / self.feature_costs, dim=1).values > self.lam
            elif self.mode == 'budget':
                features_selected = torch.max(mask, selection)
                accept_sample = torch.sum(features_selected * self.feature_costs, dim=1) <= self.budget
            elif self.mode == 'confidence':
                confidences = get_confidence(pred)
                accept_sample = confidences < self.confidence

            accept_sample = torch.bitwise_and(accept_sample, check_pos_pred_cmi)

            if sum(accept_sample).item() == 0:
                break

            mask[accept_sample] = torch.max(mask[accept_sample], selection[accept_sample])

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        if len(batch) == 2:
            return mask.cpu(), pred.cpu(), y.cpu()
        else:
            return mask.cpu(), pred.cpu()

    def format_predictions(self, outputs):
        '''Format predictions output by trainer.predict().'''
        if len(outputs[0]) == 3:
            mask_list, pred_list, y_list = zip(*outputs)
            mask = torch.cat(mask_list)
            pred = torch.cat(pred_list)
            y = torch.cat(y_list)
            return {'mask': mask, 'pred': pred, 'y': y}
        else:
            mask_list, pred_list = zip(*outputs)
            mask = torch.cat(mask_list)
            pred = torch.cat(pred_list)
            return {'mask': mask, 'pred': pred}

    def inference(self, trainer, data_loader, feature_costs=None, budget=None, lam=None, confidence=None):
        '''Make predictions on a dataset using the trained model.'''
        original_feature_costs = self.feature_costs.cpu()
        self.set_feature_costs(feature_costs)
        self.set_stopping_criterion(budget, lam, confidence)

        outputs = trainer.predict(self, data_loader)
        outputs = self.format_predictions(outputs)

        self.set_feature_costs(original_feature_costs)
        del self.mode

        return outputs