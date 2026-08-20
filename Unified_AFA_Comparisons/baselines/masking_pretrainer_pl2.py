"""
Lightning 2.x-compatible port of masking_pretrainer.py.

Two changes from the original, both purely mechanical (no logic change):
  1. training_epoch_end/validation_epoch_end (removed in Lightning 2.0)
     replaced with on_train_epoch_end/on_validation_epoch_end, manually
     accumulating step outputs as instance attributes (the migration
     path Lightning's own error message points to).
  2. ReduceLROnPlateau's verbose=True argument removed (deprecated, then
     removed, in recent PyTorch -- same bug you hit before in
     iterative.py).

Everything else -- the masking logic, the loss computation, the manual
early-stopping-via-scheduler logic -- is unchanged from your original
masking_pretrainer.py.
"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
from core.utils import generate_uniform_mask
from core.budget_state import generate_budget_constrained_mask, MaskPool


class MaskingPretrainerPL2(pl.LightningModule):
    '''
    Pretrain model with missing features. (Lightning 2.x-compatible port
    of MaskingPretrainer -- see module docstring.)

    Args: identical to the original MaskingPretrainer.
    '''

    def __init__(self,
                 model,
                 mask_layer,
                 lr,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None,
                 feature_costs=None,
                 budget_state=None):
        super().__init__()

        self.model = model
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size

        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs

        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

        # Optional budget-constrained masking (see training_step). If
        # budget_state is None (default), behavior is UNCHANGED: masks
        # are drawn uniformly at random, exactly as before.
        if budget_state is not None and feature_costs is None:
            raise ValueError('feature_costs is required when budget_state is given')
        self.feature_costs = feature_costs
        self.budget_state = budget_state
        # See MaskPool docstring in budget_state.py: epoch 0 spends
        # budget_state's total_budget exactly once (single pass over all
        # samples); every later epoch just gets a fresh permutation of
        # that same pool, paired against whatever batches the train
        # dataloader yields, with no further spending. Built lazily in
        # on_fit_start (needs budget_state to already be set).
        self.mask_pool = MaskPool(feature_costs, budget_state) if budget_state is not None else None

        self._train_step_losses = []
        self._val_step_outputs = []

    def on_fit_start(self):
        self.num_bad_epochs = 0
        # Pool reset ONCE here, at the start of the whole training run --
        # NOT per-epoch (see MaskPool docstring in budget_state.py for
        # why: epoch 0 spends total_budget exactly once via the pool,
        # and no budget is ever spent again after that).
        if self.budget_state is not None:
            self.budget_state.reset_pool()

    def on_train_epoch_start(self):
        if self.mask_pool is not None:
            self.mask_pool.start_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mask_pool is not None:
            # Epoch 0: real budget-constrained acquisition (spends
            # budget_state, updates omd_lambda). Epoch > 0: replay from
            # that epoch's permutation of the fixed epoch-0 pool -- no
            # further spending. See MaskPool in budget_state.py.
            mask = self.mask_pool.get_mask(len(x), device=x.device)
        else:
            mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        loss = self.loss_fn(pred, y)
        self._train_step_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self._train_step_losses).mean()
        self.log('Loss Train', loss, prog_bar=True, logger=True)
        self._train_step_losses.clear()

        if self.mask_pool is not None:
            self.mask_pool.end_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        self._val_step_outputs.append((pred.detach(), y.detach()))
        return pred, y

    def on_validation_epoch_end(self):
        pred_list, y_list = zip(*self._val_step_outputs)
        pred = torch.cat(pred_list)
        y = torch.cat(y_list)
        loss = self.loss_fn(pred, y)
        val_loss = self.val_loss_fn(pred, y)

        self.log('Loss Val', loss, prog_bar=True, logger=True)
        self.log('Perf Val', val_loss, prog_bar=True, logger=True)

        sch = self.lr_schedulers()
        if loss < sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            self.trainer.should_stop = True

        self._val_step_outputs.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'Loss Val'
        }
