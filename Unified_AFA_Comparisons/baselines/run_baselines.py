"""Run the EDDI and DIME baselines with the shared comparison protocol.

Both baselines always use 60/20/20 train/validation/test. For comparable
adaptive or two-stage results, use ``run_proposed_methods.py`` with
``--split-mode 60-20-20`` and the same dataset, costs, seeds, and synthetic
configuration. CAE is intentionally not included in this folder.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from core.datasets import (
    ALL_BINARY_DATASETS,
    ALL_DATASETS,
    DEFAULT_IMAGE_POOL_SIDE,
    MULTICLASS_REAL_DATASETS,
    SYNTHETIC_DATASETS,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    generate_modality_costs_heterogeneous,
    load_dataset_as_numpy,
    split_dataset,
)
from core.utils import MaskLayerGrouped, get_mlp_network, get_linear_network


def _budget_summary_fields(state, prefix):
    """
    Flatten a budget_state.BudgetState's summary() into prefixed fields
    for a results dict/CSV row -- e.g. prefix="cae_stage1" gives
    cae_stage1_lambda_final, cae_stage1_cumulative_spent,
    cae_stage1_total_budget, cae_stage1_remaining_budget_final,
    cae_stage1_spent_total.

    state=None (i.e. use_matched_budget_constraint=False, so this stage
    never had a BudgetState at all) fills every field with None, so the
    CSV always has the same columns whether or not the constraint was
    used for that trial.
    """
    if state is None:
        return {
            f"{prefix}_lambda_final": None,
            f"{prefix}_cumulative_spent": None,
            f"{prefix}_total_budget": None,
            f"{prefix}_remaining_budget_final": None,
            f"{prefix}_spent_total": None,
        }
    s = state.summary()
    return {
        f"{prefix}_lambda_final": s["lambda_final"],
        f"{prefix}_cumulative_spent": s["cumulative_spent"],
        f"{prefix}_total_budget": s["total_budget"],
        f"{prefix}_remaining_budget_final": s["remaining_budget"],
        f"{prefix}_spent_total": s["spent_total"],
    }


def f1_metric(pred, y):
    """
    F1 score, wrapped as a torch tensor so it fits BaseModel.evaluate's
    metric convention (each metric fn must return something with .item()).

    Class-count is read from pred.shape[1] (the classifier's logit width =
    d_out): exactly 2 -> binary F1 on the positive class (index 1), matching
    the original binary behavior; >2 -> MACRO-averaged F1 (unweighted mean
    of per-class F1), the standard multiclass summary. zero_division=0
    avoids a warning/error when a class is never predicted.
    """
    y_pred = pred.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    average = "binary" if pred.shape[1] == 2 else "macro"
    return torch.tensor(f1_score(y_true, y_pred, average=average, zero_division=0))


def accuracy_metric(pred, y):
    return (pred.argmax(dim=1) == y).float().mean()


def auroc_metric(pred, y):
    """
    AUROC from softmax probabilities (unlike accuracy/F1, this needs
    probabilities, not hard argmax predictions).

    Class-count is read from pred.shape[1] (= d_out): exactly 2 -> binary
    AUROC on the positive class (index 1), matching the original behavior;
    >2 -> one-vs-rest AUROC, MACRO-averaged over classes.

    roc_auc_score needs every class it scores to be present in y_true. If a
    batch/split has <2 classes present (binary), or is missing one of the K
    classes (multiclass OVR), there's no valid AUROC -- returns NaN rather
    than raising, so a sweep doesn't crash on an unlucky split.
    """
    probs = torch.softmax(pred, dim=1).detach().cpu().numpy()
    y_true = y.cpu().numpy()
    n_classes = pred.shape[1]
    if len(np.unique(y_true)) < 2:
        return torch.tensor(float("nan"))
    if n_classes == 2:
        return torch.tensor(roc_auc_score(y_true, probs[:, 1]))
    # Multiclass: OVR needs all K classes present in y_true; guard with nan.
    try:
        auroc = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        )
    except ValueError:
        auroc = float("nan")
    return torch.tensor(auroc)


# ============================================================
# Shared setup: data, split, mask layer, feature costs
# ============================================================

DEFAULT_SYNTHETIC_NVIEWS = 10


def build_experiment(dataset_name, device, data_path=None, max_modalities=None, split_seed=None,
                      synthetic_seed=SYNTHETIC_SEED,
                      nsamples=1000, n_views=None, max_samples=None, image_pool_side=None,
                      image_data_home=None, num_classes=None):
    """
    Load the chosen dataset, split into train/val/test, and build the
    shared modality structure (mask layer + feature costs) used by both
    CAE and EDDI.

    dataset_name: one of ALL_BINARY_DATASETS (real UCI datasets) OR one
    of SYNTHETIC_DATASETS (including multiclass "synthetic") -- the
    same synthetic data the GMM-bandit method itself is evaluated on
    (gmm_2class_bandit_symmetric.py / gmm_2class_bandit_asymmetric.py).
    For the synthetic datasets, data_path is ignored (nothing is read
    from disk -- see load_gmm_dataset) and nsamples controls the
    generated dataset's row count; for real datasets, nsamples is ignored.

    data_path: optional override for where to read the dataset's raw CSV
    from. Required for "miniboone"/"physionet" if your local file isn't
    at afa_tabular_datasets.DEFAULT_PATHS' default location. Ignored for
    synthetic GMM datasets.

    max_modalities: controls the number of modalities/views for BOTH
    dataset kinds, but via a different mechanism for each -- there used
    to be a separate `nviews` parameter for synthetic datasets; it's been
    folded into this one, since the two were doing the same conceptual
    job (how many modalities does this experiment use) and having both
    invited exactly the max_modalities < nviews truncation-without-
    renormalization bug this merge eliminates.
      - REAL datasets: keeps only the FIRST max_modalities columns of the
        dataset's feature matrix (in whatever column order
        load_afa_dataset returns -- i.e. the original CSV's column
        order, unchanged by preprocessing). Modality 0 (the free one) is
        always feature_names[0], so max_modalities=5 means "the free
        modality + the first 4 paid features", not 5 arbitrary/random ones.
      - SYNTHETIC datasets: GENERATES exactly max_modalities views in the
        first place (passed straight through to load_gmm_dataset's own
        nviews argument) -- there is no separate generate-then-truncate
        step, so feature_costs always sums to exactly 1 over whatever
        width was requested, with no renormalization needed. If left
        None, defaults to DEFAULT_SYNTHETIC_NVIEWS (5, matching the
        original scripts' NUM_VIEWS).

    split_seed: seed for the train/val/test split (afa_tabular_datasets.
    split_dataset's own SPLIT_SEED=42 default is used if left None).
    Vary this across a seed loop to get a genuinely different split per
    seed, not just different model initialization. For synthetic GMM
    datasets, this ALSO seeds data generation itself (see
    load_gmm_dataset) -- so unlike the real datasets (same underlying
    data, different split, across seeds), each seed here gets a
    genuinely fresh dataset AND costs, matching how the original GMM
    scripts' own per-trial loop redraws data every trial rather than
    reusing one fixed dataset. If left None, seed 42 is used (matching
    the original scripts' default SEED).

    Returns a dict with everything either runner needs.
    """
    is_synthetic = dataset_name in SYNTHETIC_DATASETS
    synthetic_n_views = n_views if n_views is not None else SYNTHETIC_N_VIEWS
    synthetic_n_classes = num_classes if num_classes is not None else SYNTHETIC_N_CLASSES
    pool = image_pool_side if image_pool_side is not None else DEFAULT_IMAGE_POOL_SIDE

    # Use the exact same loader contract and synthetic generator as the
    # aligned proposed-method runners.  The only intentional protocol
    # difference is the 60/20/20 split below, because CAE/EDDI/DIME need a
    # validation set.
    X_np, y_np, feature_names = load_dataset_as_numpy(
        dataset_name,
        max_modalities=None if is_synthetic else max_modalities,
        data_path=data_path,
        max_samples=max_samples,
        synthetic_n_samples=nsamples,
        synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed,
        synthetic_n_classes=synthetic_n_classes,
        image_pool_side=pool,
        image_data_home=image_data_home,
    )
    X = torch.as_tensor(X_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.long)
    num_modalities = X.shape[1]  # 1 free + (num_modalities - 1) paid
    d_in = num_modalities
    d_out = int(torch.unique(y).numel())
    print(f"{dataset_name}: {X.shape[0]} samples, {num_modalities} modalities "
          f"(free modality: '{feature_names[0]}', {num_modalities - 1} paid), {d_out} classes"
          + (f" [truncated to first {max_modalities}]" if (max_modalities is not None and not is_synthetic) else "")
          + (f" [generated at {num_modalities} views]" if is_synthetic else ""))

    split_kwargs = {} if split_seed is None else {"seed": split_seed}
    train_idx, val_idx, test_idx = split_dataset(X.shape[0], **split_kwargs)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_dataset = TensorDataset(X[train_idx], y[train_idx])
    val_dataset = TensorDataset(X[val_idx], y[val_idx])
    test_dataset = TensorDataset(X[test_idx], y[test_idx])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    group_matrix = torch.eye(num_modalities)
    mask_layer = MaskLayerGrouped(group_matrix).to(device)

    raw_feature_costs = np.asarray(
        generate_modality_costs_heterogeneous(
            n_features=num_modalities, dataset_name=dataset_name
        ),
        dtype=float,
    )
    feature_costs = (raw_feature_costs / raw_feature_costs.sum()).tolist()
    paid_costs = feature_costs[1:]
    print(f"feature_costs[0] (free) = {feature_costs[0]}, "
          f"paid costs: min={min(paid_costs):.4f}, max={max(paid_costs):.4f}, "
          f"mean={sum(paid_costs) / len(paid_costs):.4f}, total={sum(paid_costs):.4f}, "
          f"total (incl. free) = {sum(feature_costs):.4f}")

    return {
        "dataset_name": dataset_name,
        "split_mode": "60-20-20",
        "feature_names": feature_names,
        "num_modalities": num_modalities,
        "d_in": d_in,
        "d_out": d_out,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "mask_layer": mask_layer,
        "feature_costs": feature_costs,
        "paid_costs": paid_costs,
    }



# ============================================================
# EDDI runner (one-shot, ranked: per-feature scoring once, then a
# greedy walk down the fixed ranking -- see eddi_oneshot.py's docstring.
# NOTE: an earlier exhaustive candidate-SUBSET-scoring EDDI variant was
# removed by request; this is now the only EDDI implementation.
# ============================================================

def run_eddi(ctx, device, train_budget=None, test_budget=None,
             train_fraction=0.75, test_fraction=0.25, cost_normalized=True,
             use_matched_budget_constraint=False, lambda_max=10.0, step_size=1.0,
             linear_classifier=True):
    """
    Scores individual features once (linear in feature count), not an
    exhaustive subset powerset -- no combinatorial-blowup concern here.

    train_budget/test_budget: absolute cost ceilings, if you want to set
    them directly. If left None, they default to train_fraction/
    test_fraction of the TOTAL paid cost -- matching your original
    toy_multimodal.csv convention (train_budget=0.75, test_budget=0.25
    when feature_costs summed to 1.0 across paid features).

    use_matched_budget_constraint: same idea as run_dime's flag -- see
    that function's docstring for the full rationale. Here it applies to
    BOTH training stages that actually touch masking (PVAE's sampler.fit
    and IterativeSelector.fit, which trains the predictor EDDI's scoring
    depends on), plus a global depleting pool at inference (both the
    train-side and test-side select_features calls). Default False
    preserves the ORIGINAL per-sample-budget behavior exactly.

    linear_classifier: DEFAULT True. If True, the PREDICTOR (the model EDDI's per-
    feature scoring actually queries, trained via IterativeSelector) uses
    a plain linear classifier (get_linear_network) instead of the
    default nonlinear MLP. The PVAE sampler's encoder/decoder are left
    as nonlinear MLPs regardless -- PVAE is a generative imputation
    model, not the classifier being compared, and forcing it linear
    would confound imputation quality with classifier-capacity fairness
    rather than isolating it. See run_cae's linear_classifier docstring
    for the broader rationale.
    """
    classifier_fn = get_linear_network if linear_classifier else get_mlp_network

    from baselines.eddi_oneshot import EDDI
    from baselines.iterative import IterativeSelector, UniformSampler
    from core.budget_state import BudgetState
    from baselines import pvae

    feature_names = ctx["feature_names"]
    num_modalities = ctx["num_modalities"]
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    train_dataset = ctx["train_dataset"]
    test_dataset = ctx["test_dataset"]
    train_dataloader = ctx["train_dataloader"]
    val_dataloader = ctx["val_dataloader"]
    mask_layer = ctx["mask_layer"]
    feature_costs = ctx["feature_costs"]
    paid_costs = ctx["paid_costs"]
    total_paid_cost = sum(paid_costs)

    if train_budget is None:
        train_budget = train_fraction * total_paid_cost
    if test_budget is None:
        test_budget = test_fraction * total_paid_cost

    print(f"\n=== EDDI === total_paid_cost={total_paid_cost:.4f}, "
          f"train_budget={train_budget:.4f} ({train_budget / total_paid_cost:.2%} of total), "
          f"test_budget={test_budget:.4f} ({test_budget / total_paid_cost:.2%} of total), "
          f"use_matched_budget_constraint={use_matched_budget_constraint}")

    n_train = len(train_dataset)
    n_test = len(test_dataset)
    feature_costs_tensor = torch.tensor(feature_costs, dtype=torch.float32)

    pvae_budget_state = None
    predictor_budget_state = None
    train_inference_budget_state = None
    test_inference_budget_state = None
    if use_matched_budget_constraint:
        pvae_budget_state = BudgetState(spending_ratio=train_budget, total_budget=train_budget * n_train,
                                         lambda_max=lambda_max, step_size=step_size)
        predictor_budget_state = BudgetState(spending_ratio=train_budget, total_budget=train_budget * n_train,
                                              lambda_max=lambda_max, step_size=step_size)
        train_inference_budget_state = BudgetState(spending_ratio=train_budget, total_budget=train_budget * n_train,
                                                     lambda_max=lambda_max, step_size=step_size)
        test_inference_budget_state = BudgetState(spending_ratio=test_budget, total_budget=test_budget * n_test,
                                                    lambda_max=lambda_max, step_size=step_size)

    # ------------------------------------------------------
    # Train PVAE sampler
    # ------------------------------------------------------
    num_groups = num_modalities
    bottleneck = 16
    encoder = get_mlp_network(d_in + num_groups, bottleneck * 2)
    decoder = get_mlp_network(bottleneck, d_in)
    sampler = pvae.PVAE(encoder, decoder, mask_layer, num_samples=128, decoder_distribution='gaussian').to(device)
    pvae_t0 = time.time()
    sampler.fit(train_dataloader, val_dataloader, lr=1e-3, nepochs=50, verbose=False,
                feature_costs=feature_costs_tensor if use_matched_budget_constraint else None,
                budget_state=pvae_budget_state)
    pvae_train_time = time.time() - pvae_t0

    # ------------------------------------------------------
    # Train predictor under uniform random masking
    # ------------------------------------------------------
    model = classifier_fn(d_in + num_groups, d_out).to(device)
    mask_sampler = UniformSampler(train_dataset.tensors[0])
    iterative_selector = IterativeSelector(model, mask_layer, mask_sampler).to(device)
    predictor_t0 = time.time()
    iterative_selector.fit(train_dataloader, val_dataloader, lr=1e-3, nepochs=50,
                            loss_fn=nn.CrossEntropyLoss(), patience=5, verbose=False,
                            feature_costs=feature_costs_tensor if use_matched_budget_constraint else None,
                            budget_state=predictor_budget_state)
    predictor_train_time = time.time() - predictor_t0
    train_time_sec = pvae_train_time + predictor_train_time
    print(f"EDDI -- PVAE train time: {pvae_train_time:.1f}s | predictor train time: {predictor_train_time:.1f}s | "
          f"train_time_sec: {train_time_sec:.1f}s")

    eddi_model = EDDI(sampler, model, mask_layer, task='classification',
                       feature_costs=feature_costs, cost_normalized=cost_normalized).to(device)
    print(f"EDDI cost_normalized={cost_normalized} "
          f"({'gain-per-cost' if cost_normalized else 'raw gain under hard budget'})")

    # Training-side selection
    X_train = train_dataset.tensors[0].to(device)
    x_train_masked, m_train, train_cost = eddi_model.select_features(
        X_train, budget=train_budget, verbose=False, global_budget_state=train_inference_budget_state)
    print(f"EDDI -- train avg cost/sample: {train_cost / len(X_train):.4f}")

    train_selection_freq = m_train.mean(dim=0).cpu()
    train_freq_by_name = {
        feature_names[i]: float(train_selection_freq[i]) for i in range(num_modalities)
    }
    print("EDDI -- train-side selection frequency per feature:")
    for name, freq in sorted(train_freq_by_name.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: {freq:.2%}")

    # Inference-side selection (final evaluation, tighter budget)
    inference_t0 = time.time()
    X_test = test_dataset.tensors[0].to(device)
    y_test = test_dataset.tensors[1]
    x_test_masked, m_test, test_cost = eddi_model.select_features(
        X_test, budget=test_budget, verbose=False, global_budget_state=test_inference_budget_state)

    test_selection_freq = m_test.mean(dim=0).cpu()
    test_freq_by_name = {
        feature_names[i]: float(test_selection_freq[i]) for i in range(num_modalities)
    }
    print("EDDI -- test-side selection frequency per feature:")
    for name, freq in sorted(test_freq_by_name.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: {freq:.2%}")

    with torch.no_grad():
        pred = eddi_model.model(x_test_masked)
    inference_time = time.time() - inference_t0
    acc = (pred.argmax(dim=1).cpu() == y_test).float().mean().item()
    # Reuse the shared, multiclass-safe metric fns (binary path for K=2,
    # macro/OVR for K>2) so EDDI's numbers match CAE's and DIME's exactly.
    f1 = f1_metric(pred, y_test).item()
    auroc = auroc_metric(pred, y_test).item()
    print(f"EDDI -- test accuracy: {acc:.4f} | test F1: {f1:.4f} | test AUROC: {auroc:.4f} | "
          f"avg cost/sample: {test_cost / len(X_test):.4f}")
    print(f"EDDI -- inference time: {inference_time:.1f}s")

    return {
        "method": "eddi",
        "train_budget": train_budget,
        "test_budget": test_budget,
        "train_avg_cost_per_sample": train_cost / len(X_train),
        "test_avg_cost_per_sample": test_cost / len(X_test),
        "test_accuracy": acc,
        "test_f1": f1,
        "test_auroc": auroc,
        "pvae_train_time_sec": pvae_train_time,
        "predictor_train_time_sec": predictor_train_time,
        "train_time_sec": train_time_sec,  # = pvae_train_time_sec + predictor_train_time_sec
        "inference_time_sec": inference_time,
        "train_feature_selection_freq": train_freq_by_name,
        "test_feature_selection_freq": test_freq_by_name,
        "use_matched_budget_constraint": use_matched_budget_constraint,
        "linear_classifier": linear_classifier,
        **_budget_summary_fields(pvae_budget_state, "pvae_training"),
        **_budget_summary_fields(predictor_budget_state, "predictor_training"),
        **_budget_summary_fields(train_inference_budget_state, "train_inference"),
        **_budget_summary_fields(test_inference_budget_state, "test_inference"),
    }

# ============================================================
# DIMEOneShot runner (Option 2: value network trained FOR the one-shot
# ranking-and-commit task, not the original adaptive re-querying task)
# ============================================================
#
# Uses the Lightning-2.x-compatible files (masking_pretrainer_pl2.py,
# cmi_estimator_pl2.py, dime_oneshot_estimator.py) since the ORIGINAL
# masking_pretrainer.py/cmi_estimator.py hit a hard NotImplementedError
# on modern Lightning (validation_epoch_end/training_epoch_end were
# removed in v2.0) -- confirmed by actually running them.
#
# Pipeline (two training stages, same overall shape as before):
#   Stage A: MaskingPretrainerPL2 pretrains a predictor under random
#            masking (utils.generate_uniform_mask).
#   Stage B: DIMEOneShotEstimator (subclasses CMIEstimatorPL2, overrides
#            training_step/validation_step for Option 2 -- see that
#            file's docstring for exactly what differs and why) trains
#            the value network to directly support one-shot
#            ranking-and-commit, starting from the pretrained predictor.
#            `max_features` (a step-count cap, meaningless for a
#            budget-stopped one-shot walk) is REPLACED by `budget` (a
#            cost cap) -- mirroring exactly how max_features disappeared
#            in favor of a budget/max_budget_ever cost ceiling when your
#            original sequential EDDI was converted to eddi_oneshot.py.
#   Stage C: wrap the trained estimator in DIMEOneShot for one-shot,
#            budget-constrained inference (unchanged from before).
#
# IMPORTANT FIX: feature_costs[0] is 0 (the free modality) throughout
# this pipeline, but the ORIGINAL CMIEstimator/CMIEstimatorPL2 divide by
# self.feature_costs with NO clamping anywhere (verified: 4 raw
# `pred_cmi / self.feature_costs` divisions in your actual
# cmi_estimator.py). Passing a literal 0.0 there produces Inf, which
# argmax always "wins" -- confirmed by direct test. DIMEOneShotEstimator
# handles this internally (clamps costs, pre-marks the free modality as
# observed, matching DIMEOneShot's own convention), but nothing here
# modifies CMIEstimatorPL2/CMIEstimator's own code (per the "training
# untouched" idea for the base class) -- so this function pre-clamps the
# feature_costs TENSOR before it's ever handed to any estimator
# constructor, as a caller-side fix.

def run_dime(ctx, device, train_budget=None, test_budget=None,
             train_fraction=0.75, test_fraction=0.25, cmi_scaling="bounded",
             lr=1e-3, eps=0.05, pretrain_epochs=50, cmi_epochs=50,
             use_matched_budget_constraint=False, lambda_max=10.0, step_size=1.0,
             linear_classifier=True):
    """
    use_matched_budget_constraint: if True, training AND inference switch
    from this file's original per-sample budget conventions to the
    GMM-bandit method's matched conditions -- see budget_state.py's
    module docstring for the full rationale:
      - TRAINING (both MaskingPretrainerPL2 pretraining and
        DIMEOneShotEstimator's CMI training): a shared budget_state.
        BudgetState per stage, with `total_budget = train_budget *
        n_train` (i.e. the SAME aggregate training-side budget as if you
        ran GMM's Phase 1 over this many training rounds), enforced via
        OMD dual-ascent lambda-pricing PLUS a literal depleting pool with
        forced free-only fallback on exhaustion.
      - INFERENCE: DIMEOneShot.evaluate gets a fresh BudgetState with
        `total_budget = test_budget * n_test`, so test_budget stops being
        a per-sample cap and becomes a genuinely GLOBAL, depleting pool
        across the whole test stream (order-dependent, same as
        run_phase2_inference).
    lambda_max/step_size: FIXED calibration constants (per your
    instruction, no auto-tuning yet) -- see BudgetState's docstring.
    Default False preserves the ORIGINAL per-sample-budget behavior
    exactly (dime_estimator.budget stays a hard per-sample cap, evaluate
    uses the original per-sample budget= argument).

    linear_classifier: DEFAULT True. If True, the PREDICTOR (trained via
    MaskingPretrainerPL2, the model that actually classifies y from
    x_masked) uses a plain linear classifier (get_linear_network)
    instead of the default nonlinear MLP. The value_network (DIME's CMI/
    feature-value estimator) is left as a nonlinear MLP regardless --
    it's a scoring network, not the classifier being compared, so
    forcing it linear too would confound acquisition-scoring capacity
    with classifier-capacity fairness rather than isolating it. See
    run_cae's linear_classifier docstring for the broader rationale.
    """
    classifier_fn = get_linear_network if linear_classifier else get_mlp_network

    from baselines.dime_oneshot import DIMEOneShot
    from baselines.dime_oneshot_estimator import DIMEOneShotEstimator
    from baselines.masking_pretrainer_pl2 import MaskingPretrainerPL2
    from core.budget_state import BudgetState
    import pytorch_lightning as pl

    num_modalities = ctx["num_modalities"]
    d_in = ctx["d_in"]
    d_out = ctx["d_out"]
    train_dataloader = ctx["train_dataloader"]
    val_dataloader = ctx["val_dataloader"]
    train_dataset = ctx["train_dataset"]
    test_dataset = ctx["test_dataset"]
    mask_layer = ctx["mask_layer"]
    feature_costs = ctx["feature_costs"]
    paid_costs = ctx["paid_costs"]
    total_paid_cost = sum(paid_costs)

    if train_budget is None:
        train_budget = train_fraction * total_paid_cost
    if test_budget is None:
        test_budget = test_fraction * total_paid_cost

    print(f"\n=== DIME (one-shot, Option 2 training) === "
          f"train_budget={train_budget:.4f}, test_budget={test_budget:.4f}, eps={eps}, "
          f"use_matched_budget_constraint={use_matched_budget_constraint}")

    # MaskingPretrainerPL2 uses Lightning's automatic backward, which needs
    # a SCALAR loss -- reduction='mean' (default). DIMEOneShotEstimator
    # (like the original CMIEstimator) does its OWN manual reduction
    # (explicit .mean() calls) on a PER-SAMPLE loss, needed for the
    # per-sample CMI delta computation -- reduction='none'. Conflating
    # these two crashes with "too many indices for tensor of dimension 0"
    # inside the CMI delta computation (confirmed by actually running this).
    pretrain_loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    dime_loss_fn = nn.CrossEntropyLoss(reduction='none')
    # Clamp BEFORE handing to any estimator constructor -- see caveat above.
    feature_costs_tensor = torch.clamp(torch.tensor(feature_costs, dtype=torch.float32), min=1e-12)

    # pl.Trainer does NOT infer the accelerator from a model already moved
    # via .to(device) -- left unset, it silently trains on CPU even when
    # device is cuda (confirmed by actually running this: Lightning prints
    # "GPU available but not used" and GPU utilization stays at 0%). Derive
    # accelerator/devices explicitly from the SAME device the rest of this
    # function uses, so Stage A/B actually run where CAE/EDDI's manual
    # training loops already do.
    pl_accelerator = "gpu" if device.type == "cuda" else "cpu"
    pl_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else 1

    pretrain_budget_state = None
    cmi_budget_state = None
    inference_budget_state = None
    if use_matched_budget_constraint:
        n_train = len(train_dataset)
        n_test = len(test_dataset)
        # spending_ratio matches this stage's own per-sample budget
        # convention; total_budget scales that up to the aggregate pool
        # for the number of rounds/samples this stage actually sees.
        pretrain_budget_state = BudgetState(
            spending_ratio=train_budget, total_budget=train_budget * n_train,
            lambda_max=lambda_max, step_size=step_size)
        cmi_budget_state = BudgetState(
            spending_ratio=train_budget, total_budget=train_budget * n_train,
            lambda_max=lambda_max, step_size=step_size)
        inference_budget_state = BudgetState(
            spending_ratio=test_budget, total_budget=test_budget * n_test,
            lambda_max=lambda_max, step_size=step_size)

    # --- Stage A: pretrain predictor under random masking ------------
    predictor = classifier_fn(d_in + num_modalities, d_out).to(device)
    pretrainer = MaskingPretrainerPL2(
        predictor, mask_layer, lr=lr, loss_fn=pretrain_loss_fn, val_loss_fn=val_loss_fn,
        feature_costs=feature_costs_tensor if use_matched_budget_constraint else None,
        budget_state=pretrain_budget_state)

    t0 = time.time()
    trainer_a = pl.Trainer(max_epochs=pretrain_epochs, enable_progress_bar=False, enable_checkpointing=False,
                            accelerator=pl_accelerator, devices=pl_devices, logger=False)
    trainer_a.fit(pretrainer, train_dataloader, val_dataloader)
    pretrain_time = time.time() - t0
    print(f"MaskingPretrainerPL2 pretraining took {pretrain_time:.1f}s")

    # --- Stage B: train DIMEOneShotEstimator, from the pretrained predictor ---
    value_network = get_mlp_network(d_in + num_modalities, num_modalities).to(device)
    dime_estimator = DIMEOneShotEstimator(
        value_network=value_network,
        predictor=pretrainer.model,
        mask_layer=mask_layer,
        lr=lr,
        budget=train_budget,
        eps=eps,
        loss_fn=dime_loss_fn,
        val_loss_fn=val_loss_fn,
        feature_costs=feature_costs_tensor,
        cmi_scaling=cmi_scaling,
        budget_state=cmi_budget_state,
    ).to(device)
    print(f"DIMEOneShotEstimator derived max_walk_depth={dime_estimator.max_features} from train_budget"
          + (" (efficiency bound only -- real stopping rule is the global pool under "
             "use_matched_budget_constraint)" if use_matched_budget_constraint else ""))

    t0 = time.time()
    trainer_b = pl.Trainer(max_epochs=cmi_epochs, enable_progress_bar=False, enable_checkpointing=False,
                            accelerator=pl_accelerator, devices=pl_devices, logger=False)
    trainer_b.fit(dime_estimator, train_dataloader, val_dataloader)
    cmi_train_time = time.time() - t0
    print(f"DIMEOneShotEstimator training took {cmi_train_time:.1f}s")

    # --- Stage C: one-shot wrapper + budgeted evaluation -------------
    dime_model = DIMEOneShot(dime_estimator, feature_costs=feature_costs).to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    inference_t0 = time.time()
    metrics, avg_cost = dime_model.evaluate(
        test_dataloader,
        {"accuracy": accuracy_metric, "f1": f1_metric, "auroc": auroc_metric},
        budget=test_budget,
        global_budget_state=inference_budget_state,
    )
    inference_time = time.time() - inference_t0
    train_time_sec = pretrain_time + cmi_train_time
    print(f"DIME -- test accuracy: {metrics['accuracy']:.4f} | test F1: {metrics['f1']:.4f} | "
          f"test AUROC: {metrics['auroc']:.4f} | avg cost/sample: {avg_cost:.4f}")
    print(f"DIME -- train time: {train_time_sec:.1f}s | inference time: {inference_time:.1f}s | "
          f"total time: {train_time_sec + inference_time:.1f}s")

    return {
        "method": "dime",
        "train_budget": train_budget,
        "test_budget": test_budget,
        "max_walk_depth": dime_estimator.max_features,
        "test_accuracy": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_auroc": metrics["auroc"],
        "test_avg_cost_per_sample": avg_cost,
        "pretrain_time_sec": pretrain_time,
        "cmi_train_time_sec": cmi_train_time,
        "train_time_sec": train_time_sec,  # = pretrain_time_sec + cmi_train_time_sec
        "inference_time_sec": inference_time,
        "use_matched_budget_constraint": use_matched_budget_constraint,
        "linear_classifier": linear_classifier,
        **_budget_summary_fields(pretrain_budget_state, "pretrain_training"),
        **_budget_summary_fields(cmi_budget_state, "cmi_training"),
        **_budget_summary_fields(inference_budget_state, "test_inference"),
    }


# ============================================================
# Budget-fraction sweep (overall budget_fraction, train/inference split
# auto-derived from actual train/test sample counts -- see docstring)
# ============================================================

# ============================================================
# Budget-fraction sweep (overall budget_fraction, train/inference split
# auto-derived from actual train/test sample counts -- see docstring)
# ============================================================

def run_budget_sweep(ctx, device, method, budget_fractions, eddi_cost_normalized=True,
                      seed=None, use_matched_budget_constraint=True,
                      lambda_max=10.0, step_size=1.0, linear_classifier=True):
    """
    Sweep over a list of overall budget fractions.

    total_budget = budget_fraction * N_total * total_paid_cost

    where N_total is the number of samples that actually go through
    budget-constrained feature acquisition -- i.e. train + test ONLY,
    excluding val (val is only ever used for early stopping/model
    selection inside fit(), never passed through select_features/
    CAEOneShot's budgeted selection, so it shouldn't count toward an
    acquisition-cost budget). total_paid_cost is the cost of acquiring
    every paid feature for a single sample. This is an aggregate,
    whole-dataset resource budget (e.g. "we have this much total
    acquisition budget across the samples that actually get features
    acquired"), not a per-sample number.

    That aggregate is split into a train-side total and an
    inference-side total PROPORTIONALLY TO ACTUAL SAMPLE COUNTS
    (train_inference_split is auto-derived as n_train/(n_train+n_test),
    not a fixed value you pass in) -- this is what makes per-sample
    train_budget and test_budget come out EXACTLY EQUAL to
    budget_fraction * total_paid_cost, with no mismatch and no need to
    cap anything:
        train_inference_split = n_train / (n_train + n_test)
        train_total = train_inference_split * total_budget
        test_total  = (1 - train_inference_split) * total_budget
        train_budget (per-sample) = train_total / n_train  == budget_fraction * total_paid_cost
        test_budget  (per-sample) = test_total  / n_test   == budget_fraction * total_paid_cost

    method: "cae", "eddi", or "dime" (run one at a time; call once per method).
    use_matched_budget_constraint: default True -- ALL methods train under
    the OMD/global-depleting-pool mechanism (see run_dime's docstring for
    the full rationale) unless explicitly disabled.
    Returns a list of dicts, one per budget_fraction, with all the
    intermediate quantities for logging/inspection.
    """
    total_paid_cost = sum(ctx["paid_costs"])
    n_train = len(ctx["train_dataset"])
    n_test = len(ctx["test_dataset"])
    n_total = n_train + n_test  # val excluded -- see docstring
    train_inference_split = n_train / n_total  # auto-derived, matches actual sample proportions

    results = []

    for budget_fraction in budget_fractions:
        total_budget = budget_fraction * n_total * total_paid_cost
        train_total = train_inference_split * total_budget
        test_total = (1 - train_inference_split) * total_budget

        train_budget = train_total / n_train
        test_budget = test_total / n_test

        print(f"\n{'#' * 60}")
        print(f"# budget_fraction={budget_fraction:.2f} -> "
              f"total_budget={total_budget:.4f} (N_total={n_total} [train+test, val excluded], "
              f"total_paid_cost={total_paid_cost:.4f})")
        print(f"#   train_inference_split={train_inference_split:.4f} (= n_train/{n_total} = {n_train}/{n_total})")
        print(f"#   train_total={train_total:.4f} / n_train={n_train} -> per-sample train_budget={train_budget:.4f}")
        print(f"#   test_total={test_total:.4f} / n_test={n_test} -> per-sample test_budget={test_budget:.4f}")
        print(f"{'#' * 60}")

        trial_t0 = time.time()
        if method == "eddi":
            run_result = run_eddi(ctx, device, train_budget=train_budget, test_budget=test_budget,
                                   cost_normalized=eddi_cost_normalized,
                                   use_matched_budget_constraint=use_matched_budget_constraint,
                                   lambda_max=lambda_max, step_size=step_size,
                                   linear_classifier=linear_classifier)
        elif method == "dime":
            run_result = run_dime(ctx, device, train_budget=train_budget, test_budget=test_budget,
                                   use_matched_budget_constraint=use_matched_budget_constraint,
                                   lambda_max=lambda_max, step_size=step_size,
                                   linear_classifier=linear_classifier)
        else:
            msg = f"method must be 'eddi' or 'dime', got {method!r}"
            raise ValueError(msg)
        trial_wall_time_sec = time.time() - trial_t0

        result_row = {
            "dataset": ctx.get("dataset_name"),
            "split_mode": "60-20-20",
            "seed": seed,
            "budget_fraction": budget_fraction,
            "total_budget": total_budget,
            "train_total": train_total,
            "test_total": test_total,
            "train_budget_per_sample": train_budget,
            "test_budget_per_sample": test_budget,
            "trial_wall_time_sec": trial_wall_time_sec,
            **run_result,
        }
        results.append(result_row)

    return results


# ============================================================
# Multi-seed wrapper (repeats a full budget-fraction sweep across
# several seeds, rebuilding the data split fresh each time)
# ============================================================

def run_multi_seed_sweep(dataset_name, device, method, budget_fractions, seeds,
                          data_path=None, max_modalities=None, eddi_cost_normalized=True,
                          use_matched_budget_constraint=True, lambda_max=10.0, step_size=1.0,
                          linear_classifier=True, nsamples=1000, n_views=None, max_samples=None,
                          image_pool_side=None, image_data_home=None, num_classes=None,
                          synthetic_seed=SYNTHETIC_SEED):
    """
    Repeat run_budget_sweep across multiple seeds. Each seed:
      1. Sets torch.manual_seed(seed) -- controls model init (PVAE,
         predictor networks) and any stochastic sampling inside
         CAEOneShot/EDDI's concrete-distribution / imputation steps.
      2. Rebuilds the experiment via build_experiment(..., split_seed=seed)
         -- so the train/val/test split itself is also different per
         seed, not just model initialization. This matches
         AFA-Benchmark's own convention of generating multiple dataset
         "instances" via different seeds (their default seeds:
         [42, 43, 44, 45, 46]). For the two synthetic GMM datasets, this
         seed ALSO regenerates the underlying data and costs from
         scratch (see build_experiment / load_gmm_dataset) -- each seed
         is a genuinely fresh dataset, matching the original GMM
         scripts' own per-trial data regeneration.

    use_matched_budget_constraint: default True -- see run_budget_sweep /
    run_dime docstrings for the full rationale.
    linear_classifier: default True -- see run_cae's docstring for the
    full rationale (isolating acquisition strategy from classifier
    capacity). Pass False to use the nonlinear MLP classifier instead.
    nsamples: only used for synthetic GMM datasets (see build_experiment);
    ignored for real datasets. View count is controlled by max_modalities
    for both dataset kinds -- see build_experiment's docstring.

    Returns a list of result dicts (same shape as run_budget_sweep's
    output) with an added "seed" field, across all seeds and budget
    fractions -- i.e. len(seeds) * len(budget_fractions) rows.
    """
    all_results = []
    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"=== SEED {seed} ===")
        print(f"{'=' * 70}")

        torch.manual_seed(seed)
        ctx = build_experiment(dataset_name, device, data_path=data_path,
                                max_modalities=max_modalities, split_seed=seed,
                                synthetic_seed=synthetic_seed,
                                nsamples=nsamples, n_views=n_views, max_samples=max_samples,
                                image_pool_side=image_pool_side,
                                image_data_home=image_data_home, num_classes=num_classes)

        seed_results = run_budget_sweep(ctx, device, method, budget_fractions,
                                         eddi_cost_normalized=eddi_cost_normalized, seed=seed,
                                         use_matched_budget_constraint=use_matched_budget_constraint,
                                         lambda_max=lambda_max, step_size=step_size,
                                         linear_classifier=linear_classifier)
        all_results.extend(seed_results)

    return all_results


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run EDDI/DIME baselines on an AFA-Benchmark dataset (binary or multiclass).")
    parser.add_argument("--dataset",
                         choices=ALL_DATASETS,
                         default="ckd",
                         help="Real binary UCI datasets: " + ", ".join(ALL_BINARY_DATASETS) + ". "
                              "Real multiclass datasets (diabetes: 3-class local CSV; mnist/"
                              "fashion_mnist: 10-class, fetched via OpenML, image pixels as "
                              "tabular features): " + ", ".join(MULTICLASS_REAL_DATASETS) + ". "
                              "Synthetic datasets (regenerated fresh per seed -- see "
                              "the shared synthetic generators; synthetic is the K-class one, K set by "
                              "--num-classes, width by --n-views): " + ", ".join(SYNTHETIC_DATASETS) + ". "
                              "Same registry the submodular runner accepts.")
    parser.add_argument("--data-path", type=str, default=None,
                         help="Override path to the dataset's raw CSV "
                              "(required for miniboone/physionet if not "
                              "at the default location).")
    parser.add_argument("--max-modalities", type=str, default="all",
                         help="'all' (default) or an integer, matching the submodular runner. "
                              "For REAL datasets: 'all' keeps every native feature column; an "
                              "integer keeps only the first N (modality 0 = free feature + first "
                              "N-1 paid). For SYNTHETIC datasets it has NO effect -- the generated "
                              "view count is set by --n-views (as in the submodular runner).")
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS,
                         help="Synthetic datasets only: number of views/modalities to GENERATE "
                              f"(default {SYNTHETIC_N_VIEWS}). Ignored for real datasets. Match "
                              "the submodular runner's --n-views for a like-for-like synthetic "
                              "comparison.")
    parser.add_argument("--n-samples", "--nsamples", dest="nsamples", type=int, default=1000,
                         help="Synthetic datasets only: rows to generate (default 1000). Ignored "
                              "for real datasets (use --max-samples to cap those). Match the "
                              "submodular runner's --n-samples. (--nsamples is kept as an alias.)")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED,
                         help="Synthetic data-generation seed, independent of --seeds. "
                              "Use the same value for adaptive, two-stage, EDDI, and DIME.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap a REAL dataset to at most this many rows (reproducible "
                              "subsample with a fixed seed, so it's stable across the seed "
                              "sweep). Chiefly for the 70k-row mnist/fashion_mnist. Ignored "
                              "for synthetic datasets (use --nsamples there). To compare "
                              "against the proposed-method runner, pass it the SAME value.")
    parser.add_argument("--num-classes", type=int, default=None,
                         help="synthetic only: how many classes K to generate (default "
                              f"{SYNTHETIC_N_CLASSES}). Ignored for every other dataset (real "
                              "datasets and the binary synthetic generators fix their own K). "
                              "Match the proposed runner's --num-classes.")
    parser.add_argument("--image-pool-side", type=int, default=None,
                         help="mnist/fashion_mnist only: block-average each 28x28 image down "
                              f"to x*x features (default {DEFAULT_IMAGE_POOL_SIDE} -> "
                              f"{DEFAULT_IMAGE_POOL_SIDE**2} features). Pass 28 to keep all 784 "
                              "raw pixels. Ignored for non-image datasets. Match the proposed "
                              "runner's --image-pool-side.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                         help="mnist/fashion_mnist only: directory fetch_openml caches its "
                              "download in. Default (None) resolves to core.datasets."
                              "DEFAULT_OPENML_DATA_HOME (a relative 'data/openml_cache' folder, "
                              "NOT fetch_openml's ~/scikit_learn_data, whose $HOME quota is tiny "
                              "on clusters like NCI Gadi). Ignored for non-image datasets.")
    parser.add_argument("--method", choices=["eddi", "dime", "all"], default="all",
                         help="'all' = eddi+dime.")
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9",
                         help="Comma-separated list of overall budget fractions to sweep, "
                              "e.g. '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9'. If set, runs "
                              "a multi-seed budget sweep instead of a single run. "
                              "Train/inference split is auto-derived from actual "
                              "train/test sample counts (not a fixed ratio) so per-sample "
                              "train_budget and test_budget always come out equal to "
                              "budget_fraction * total_paid_cost -- see run_budget_sweep docstring.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51",
                         help="Comma-separated list of seeds to repeat the budget-fraction "
                              "sweep across (only used when --budget-fractions is set). "
                              "Each seed gets its own train/val/test split AND its own "
                              "model initialization -- see run_multi_seed_sweep docstring. "
                              "Default matches AFA-Benchmark's own default instance seeds.")
    parser.add_argument("--eddi-no-cost-normalization", action="store_true",
                         help="Turn OFF EDDI's cost-normalized scoring (score / subset_cost). "
                              "By default EDDI scores by gain-per-cost, which makes it favor "
                              "cheap subsets and stay roughly budget-invariant. With this flag, "
                              "EDDI scores by RAW information gain under the hard budget filter, "
                              "so it spends up to the budget for accuracy-at-budget comparisons. "
                              "'accuracy at budget B'. See EDDI.__init__ docstring.")
    parser.add_argument("--output-csv", type=str, default=None,
                         help="Path to save results (accuracy, cost, budgets, etc.) to. If not "
                              "given, auto-generated as "
                              "'results_{method}_{dataset}_max{max_modalities or ALL}_seeds{n_seeds}.csv' "
                              "in the current directory -- e.g. --max-modalities not passed and "
                              "5 seeds swept gives 'results_dime_ckd_maxALL_seeds5.csv'. Pass this "
                              "flag to override with your own path instead.")
    parser.add_argument("--legacy-training", action="store_true",
                         help="Disable the matched-budget-constraint training (OMD lambda-pricing "
                              "+ global depleting pool + per-epoch pool reset -- see budget_state.py "
                              "and run_dime's docstring). Default is OFF, meaning ALL methods "
                              "(EDDI/DIME) train under the matched-conditions mechanism by "
                              "default. Pass this flag to fall back to each baseline's original, "
                              "unmodified training behavior instead.")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                         help="OMD dual-price ceiling (fixed calibration constant, not auto-tuned). "
                              "Only affects runs using the matched-budget-constraint training.")
    parser.add_argument("--step-size", type=float, default=1.0,
                         help="OMD dual-ascent step size (fixed calibration constant, not "
                              "auto-tuned). Only affects runs using the matched-budget-constraint "
                              "training.")
    parser.add_argument("--mlp-classifier", action="store_false", dest="linear_classifier", default=True,
                         help="Use the default 2-hidden-layer nonlinear MLP classifier instead "
                              "of the (now DEFAULT) plain linear classifier (single affine "
                              "layer, logistic/softmax regression), for the actual label-"
                              "predicting classifier in each method (EDDI's predictor and "
                              "DIME's predictor). Scoring/generative networks "
                              "that AREN'T the classifier -- DIME's value_network, EDDI's PVAE "
                              "encoder/decoder -- are left as nonlinear MLPs regardless. Linear "
                              "is the default so acquisition-strategy quality isn't confounded "
                              "with classifier-capacity advantage.")
    args = parser.parse_args()

    # --max-modalities is 'all' or an integer (str), matching the submodular
    # runner. Normalize to None (= all / no truncation) or int here, so the
    # rest of the code keeps working with the original None/int contract.
    max_modalities = None if str(args.max_modalities).lower() == "all" else int(args.max_modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eddi_cost_normalized = not args.eddi_no_cost_normalization
    use_matched_budget_constraint = not args.legacy_training

    all_results = []

    if args.budget_fractions is not None:
        budget_fractions = [float(x) for x in args.budget_fractions.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        n_seeds_for_name = len(seeds)  # actually swept -- see filename note below
        if args.method == "all":
            methods = ["eddi", "dime"]
        else:
            methods = [args.method]
        for method in methods:
            all_results.extend(run_multi_seed_sweep(
                args.dataset, device, method, budget_fractions, seeds,
                data_path=args.data_path, max_modalities=max_modalities,
                eddi_cost_normalized=eddi_cost_normalized,
                use_matched_budget_constraint=use_matched_budget_constraint,
                lambda_max=args.lambda_max, step_size=args.step_size,
                linear_classifier=args.linear_classifier,
                nsamples=args.nsamples, n_views=args.n_views, max_samples=args.max_samples,
                synthetic_seed=args.synthetic_seed,
                image_pool_side=args.image_pool_side,
                image_data_home=args.image_cache_dir, num_classes=args.num_classes,
            ))
    else:
        n_seeds_for_name = 1  # single run, no seed sweep -- see filename note below
        ctx = build_experiment(args.dataset, device, data_path=args.data_path, max_modalities=max_modalities,
                                synthetic_seed=args.synthetic_seed,
                                nsamples=args.nsamples, n_views=args.n_views, max_samples=args.max_samples,
                                image_pool_side=args.image_pool_side,
                                image_data_home=args.image_cache_dir, num_classes=args.num_classes)

        single_run_methods = []
        if args.method in ("eddi", "all"):
            single_run_methods.append(("eddi", lambda c, d: run_eddi(
                c, d, cost_normalized=eddi_cost_normalized,
                use_matched_budget_constraint=use_matched_budget_constraint,
                lambda_max=args.lambda_max, step_size=args.step_size,
                linear_classifier=args.linear_classifier)))
        if args.method in ("dime", "all"):
            single_run_methods.append(("dime", lambda c, d: run_dime(
                c, d, use_matched_budget_constraint=use_matched_budget_constraint,
                lambda_max=args.lambda_max, step_size=args.step_size,
                linear_classifier=args.linear_classifier)))

        for method_name, run_fn in single_run_methods:
            t0 = time.time()
            result = run_fn(ctx, device)
            wall_time_sec = time.time() - t0
            row = {"dataset": args.dataset, "split_mode": "60-20-20", "seed": None,
                   "trial_wall_time_sec": wall_time_sec, **result}
            all_results.append(row)

    # Filename convention: results/results_{method}_{dataset}_max{max_modalities or ALL}_seeds{n_seeds}[_mlp].csv
    # -- {method} is args.method literally (including "all" when running every
    # method in one call). {max_modalities} is "ALL" when --max-modalities
    # wasn't passed (the full feature set was used), otherwise the integer
    # given. {n_seeds} is the number of seeds actually swept, or 1 for a
    # single (non-sweep) run. "_mlp" suffix is appended when --mlp-classifier
    # was used (i.e. the NON-default nonlinear classifier), so linear- and
    # MLP-classifier runs never silently overwrite each other's results --
    # linear is now the default, so it gets no suffix. Auto-named files are
    # placed under a "results/" subdirectory (created if missing)
    # relative to wherever the script is run from. If --output-csv is
    # given explicitly, that path is used as-is (NOT forced under results/).
    if args.output_csv:
        output_csv = args.output_csv
    else:
        from pathlib import Path
        max_modalities_str = "ALL" if max_modalities is None else str(max_modalities)
        suffix = "" if args.linear_classifier else "_mlp"
        # Synthetic runs: tag the class count so K-sweeps don't overwrite each
        # other (real datasets' K is fixed by the data, so no tag). Mirrors the
        # submodular runner's _classes{K} filename tag.
        classes_tag = f"_classes{args.num_classes}" if args.dataset in SYNTHETIC_DATASETS else ""
        filename = f"results_{args.method}_{args.dataset}_max{max_modalities_str}_seeds{n_seeds_for_name}{suffix}{classes_tag}.csv"
        output_csv = str(Path("results") / filename)

    import pandas as pd
    from pathlib import Path
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # selected_feature_names/selected_modalities are lists -- stringify for a flat CSV
    flat_results = []
    for r in all_results:
        flat = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
        flat_results.append(flat)
    pd.DataFrame(flat_results).to_csv(out_path, index=False)
    print(f"\nSaved {len(all_results)} result row(s) to {out_path}")


if __name__ == "__main__":
    main()
