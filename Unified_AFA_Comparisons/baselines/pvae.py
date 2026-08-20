# Got a couple tricks from: https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py

import torch
import torch.nn as nn
import torch.optim as optim
from core.utils import generate_uniform_mask, restore_parameters
from core.budget_state import generate_budget_constrained_mask, MaskPool
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm
from copy import deepcopy


class PVAE(nn.Module):
    '''
    Partial VAE (PVAE): a variational autoencoder trained with random feature subsets.
    
    Original paper: https://arxiv.org/abs/1809.11142v4
    
    Args:
      encoder: encoder network.
      decoder: decoder network.
      mask_layer: layer to perform masking on encoder input.
      num_samples: number of latent variable samples to use during training.
      decoder_distribution: distribution for reconstruction, 'gaussian' or
        'bernoulli'.
      deterministic_kl: calculate prior/posterior KL divergence
        deterministically or stochastically.
    '''

    def __init__(self,
                 encoder,
                 decoder,
                 mask_layer,
                 num_samples=128,
                 decoder_distribution='gaussian',
                 deterministic_kl=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_layer = mask_layer
        self.num_samples = num_samples
        self.deterministic_kl = deterministic_kl
        
        assert decoder_distribution in ('gaussian', 'bernoulli')
        self.decoder_distribution = decoder_distribution
    
    def forward(self, x, mask):
        # Get latent encoding.
        x_masked = self.mask_layer(x, mask)
        latent = self.encoder(x_masked)
        
        # Sample latent variables and decode.
        dims = latent.shape[1] // 2
        mean = latent[:, :dims]
        std = torch.exp(latent[:, dims:])
        eps = torch.randn(mean.shape[0], self.num_samples, mean.shape[1], device=mean.device)
        z = torch.unsqueeze(mean, 1) + eps * torch.unsqueeze(std, 1)
    
        # Decode and return.
        recon = self.decoder(z)
        return latent, z, recon
    
    def loss(self, x, mask):
        # Get latent representation and reconstruction.
        latent, z, recon = self.forward(x, mask)

        # Calculate latent KL divergence.
        latent_dims = latent.shape[1] // 2
        latent_mean = latent[:, :latent_dims]
        latent_std = torch.exp(latent[:, latent_dims:])
        if self.deterministic_kl:
            kl = torch.distributions.kl_divergence(
                Normal(latent_mean, latent_std),
                Normal(0.0, 1.0)).sum(1)
            kl = torch.unsqueeze(kl, 1)
        else:
            # Set up prior and posterior distributions.
            p_dist = Normal(0.0, 1.0)
            q_dist = Normal(latent_mean, latent_std)
            
            # Estimate KL divergence.
            log_p = p_dist.log_prob(z)
            log_q = q_dist.log_prob(z.permute(1, 0, 2)).permute(1, 0, 2)
            kl = (log_q - log_p).sum(dim=2)
        
        # Calculate output log prob.
        if self.decoder_distribution == 'gaussian':
            # TODO learned std version: unstable training
            # dims = recon.shape[2] // 2
            # mean = recon[:, :, :dims]
            # std = torch.exp(recon[:, :, dims:])
            mean = recon
            std = torch.ones_like(mean)
            dist = Normal(mean, std)
        elif self.decoder_distribution == 'bernoulli':
            p = recon.sigmoid()
            dist = Bernoulli(p)
            # x = (x > 0.5).float()  # TODO included this only for MNIST, not usually necessary
        
        log_prob = dist.log_prob(torch.unsqueeze(x, 1))
        if hasattr(self.mask_layer, 'group_matrix'):  # TODO support for groups is not elegant
            mask_multiply = mask @ self.mask_layer.group_matrix
        else:
            mask_multiply = mask
        
        log_prob = (log_prob * torch.unsqueeze(mask_multiply, 1)).sum(dim=2)
        
        # Calculate loss.
        return kl - log_prob
    
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            verbose=True,
            feature_costs=None,
            budget_state=None):
        '''
        Train model.
        
        Args:
          train_loader:
          val_loader:
          lr:
          nepochs:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          verbose:
          feature_costs: per-feature (or per-modality-group, matching
            mask_size) cost tensor/array. feature_costs[0] must be 0 (the
            free feature). REQUIRED if budget_state is given.
          budget_state: optional budget_state.BudgetState instance. If
            given (together with feature_costs), training masks are drawn
            from generate_budget_constrained_mask instead of
            generate_uniform_mask -- i.e. PVAE's own imputer training
            becomes budget-constrained via the same OMD/global-depletion
            mechanism used everywhere else, rather than seeing masks of
            uniformly-random cardinality. If None (default), behavior is
            UNCHANGED from the original: masks are drawn uniformly at
            random, exactly as before.

            NOTE: since PVAE is what EDDI's downstream subset-scoring
            (`sampler.impute`) depends on for imputation quality, biasing
            training masks toward cheap/free features here can degrade
            imputation for subsets that are comparatively expensive.
            Recommended: log validation imputation loss stratified by
            mask cardinality once this is running, so any EDDI accuracy
            change downstream can be attributed to acquisition strategy
            vs. imputer quality rather than assumed to be the former.
        '''
        if budget_state is not None and feature_costs is None:
            raise ValueError('feature_costs is required when budget_state is given')
        # Set up optimizer and lr scheduler.
        mask_layer = self.mask_layer
        device = next(self.parameters()).device
        opt = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=factor, patience=patience,
            min_lr=min_lr)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, _ = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        # For tracking best model and early stopping.
        best_encoder = None
        best_decoder = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        mask_pool = None
        if budget_state is not None:
            # Pool reset ONCE, before the epoch loop -- see MaskPool
            # docstring in budget_state.py. Mechanism: epoch 0 spends
            # budget_state's total_budget EXACTLY ONCE (a single pass
            # over every sample, budget-constrained mask per batch,
            # exactly as before); the resulting masks are cached into a
            # fixed pool. From epoch 1 onward, NO further spending
            # happens -- each epoch just gets a fresh random PERMUTATION
            # of that same pool paired against the batches coming off
            # train_loader, so PVAE still sees varied sample<->mask
            # pairings every epoch (mask diversity, which its imputation
            # role needs) without ever exceeding the budget spent once
            # in epoch 0.
            budget_state.reset_pool()
            mask_pool = MaskPool(feature_costs, budget_state)

        for epoch in tqdm(range(nepochs)):
            # Switch model to training mode.
            self.train()
            if mask_pool is not None:
                mask_pool.start_epoch(epoch)

            for x, _ in train_loader:
                # Calculate loss.
                x = x.to(device)
                if mask_pool is not None:
                    m = mask_pool.get_mask(len(x), device=device)
                else:
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                loss = self.loss(x, m).mean()

                # Take gradient step.
                loss.backward()
                opt.step()
                self.zero_grad()

            if mask_pool is not None:
                mask_pool.end_epoch(epoch)

            # Calculate validation loss.
            self.eval()
            with torch.no_grad():
                # For mean loss.
                val_loss = 0
                n = 0

                for x, _ in val_loader:
                    # Calculate loss.
                    # TODO mask should be precomputed and shared across epochs
                    x = x.to(device)
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                    loss = self.loss(x, m).mean()
                    
                    # Update mean.
                    val_loss = (loss * len(x) + val_loss * n) / (n + len(x))
                    n += len(x)

            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}\n')
                
            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_encoder = deepcopy(self.encoder)
                best_decoder = deepcopy(self.decoder)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                
            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch+1}')
                break

        # Copy parameters from best model.
        restore_parameters(self.encoder, best_encoder)
        restore_parameters(self.decoder, best_decoder)
    
    def impute(self, x, mask):
        '''Impute using a partial input.'''
        _, _, recon = self.forward(x, mask)
        return self.output_sample(recon)
    
    def generate(self, num_samples):
        '''Generate new samples by sampling from the latent distribution.'''
        dim = list(self.decoder.parameters())[0].shape[1]
        device = next(self.decoder.parameters()).device
        z = torch.randn(num_samples, dim, device=device)
        
        # Decode.
        recon = self.decoder(z)
        return self.output_sample(recon)
        
    def output_sample(self, params):
        '''Generate output sample given decoder parameters.'''
        if self.decoder_distribution == 'gaussian':
            # Return mean.
            mean = params
            return mean

        elif self.decoder_distribution == 'bernoulli':
            # Return probabilities.
            p = params.sigmoid()
            return p