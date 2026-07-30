"""
two_stage_multiclass_greedy_runner.py

Driver for two_stage_multiclass_greedy.py. A near-clone of
two_stage_multiclass_runner.py -- same data loading, same cost model, same
80/20 split, same seeds x budget_fractions x init_fractions sweep, same row
schema (so run_proposed_methods.normalize_two_stage_mc consumes it
unchanged), same Excel output shape.

=== Differences from two_stage_multiclass_runner.py ===
1. NO expert enumeration. generate_view_combinations /
   generate_combination_costs_heterogeneous are gone, and with them the
   2^(nviews-1) blow-up: --max-modalities defaults to "all" here and the
   MAX_RECOMMENDED_MODALITIES warning is dropped. The per-view `costs`
   array the LP inference already needed is now ALSO what Stage-2 training
   consumes. (nviews is still bounded in practice by what the Stage-2 LP
   branch-and-bound pricing can handle -- the same regime submodular
   runs in.)
2. --warm-start and --gamma-max are gone (no expert weight vector, no
   EXP4 exploration rate). --step-size / --lambda-max / --pred-rule are
   unchanged -- the OMD dual and the decision rule are shared.
3. New: --center-update {frozen,reward_full,reward_bandit,full,bandit},
   --alpha-ucb, --lr. There is deliberately NO warmup flag: Stage 1 IS the
   warmup, and a second one measured harmful. See
   docstring, and note the comparability warning there for full/bandit.
4. Stage-2 INFERENCE uses the centres run_alg_greedy_multiclass RETURNS,
   not the Stage-1 ones. Under center_update="frozen"/"counts" those are
   identical objects by value, so this is a no-op there; under
   "full"/"bandit" it is the whole point (the acquisition-time updates
   have to reach the classifier used at inference, exactly as in
   gmm_multiclass_submodular_runner.py).
5. Extra columns: 'center_update', 'avg_views_acquired', 'n_unique_masks',
   'Experts' is emitted as NaN to keep the column
   present and the two runners' frames concatenable.

Label matching, budget derivation, and the "Total Reward" convention are
IDENTICAL to two_stage_multiclass_runner.py -- see that module's docstring.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from core.datasets import (
    ALL_DATASETS,
    DEFAULT_IMAGE_POOL_SIDE,
    MULTICLASS_SYNTHETIC_DATASETS,
    SYNTHETIC_CLUSTER_STD,
    SYNTHETIC_DATASETS,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_SAMPLES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    generate_modality_costs_heterogeneous,
    load_dataset_as_numpy,
    split_train_inference,
)
from core.excel_utils import _style_sheet

from two_stage.two_stage_multiclass import (
    PRED_RULES,
    calculate_two_stage_error,
    initialize_centers_multiclass,
    run_inference_lp_dataset_colgen_multiclass,
)
from two_stage_greedy.two_stage_multiclass_greedy import (
    CENTER_UPDATE_MODES,
    REWARD_UPDATE_SCOPES,
    run_alg_greedy_multiclass,
)


def run_experiment(
    dataset_name,
    max_modalities=None,
    seeds=(42, 43, 44, 45, 46),
    budget_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    n_init_fraction_points=10,
    data_path=None,
    max_samples=None,
    synthetic_n_samples=SYNTHETIC_N_SAMPLES,
    synthetic_n_views=SYNTHETIC_N_VIEWS,
    synthetic_seed=SYNTHETIC_SEED,
    synthetic_mean_scale=SYNTHETIC_MEAN_SCALE,
    synthetic_cluster_std=SYNTHETIC_CLUSTER_STD,
    synthetic_n_classes=SYNTHETIC_N_CLASSES,
    center_update="reward_full",
    reward_update="subsets",
    alpha_ucb=2.0,
    lr=1e-2,
    step_size=1.0,
    lambda_max=10.0,
    run_inference=True,
    image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
    image_data_home=None,
    pred_rule="nearest_center",
):
    """Greedy-Stage-2 counterpart of
    two_stage_multiclass_runner.run_experiment. Identical sweep structure
    and row schema; see this module's docstring for the parameter deltas
    and two_stage_multiclass_greedy.run_alg_greedy_multiclass's docstring
    for center_update / alpha_ucb / lr semantics.

    run_inference: if False, Stage-2 LP inference is skipped, all
        inference_* columns become NaN and Total Reward becomes the
        train-only sample-weighted accuracy -- exactly as in the EXP4
        runner, so --skip-inference rows remain comparable BETWEEN the two
        methods (but not against a full run of either).
    """
    if dataset_name not in ALL_DATASETS:
        msg = f"dataset_name must be one of {ALL_DATASETS}, got {dataset_name!r}"
        raise ValueError(msg)
    if center_update not in CENTER_UPDATE_MODES:
        raise ValueError(f"center_update must be one of {CENTER_UPDATE_MODES}, "
                         f"got {center_update!r}")

    X_full, Y_full, feature_names = load_dataset_as_numpy(
        dataset_name, max_modalities=max_modalities, data_path=data_path,
        max_samples=max_samples, synthetic_n_samples=synthetic_n_samples,
        synthetic_n_views=synthetic_n_views, synthetic_seed=synthetic_seed,
        synthetic_mean_scale=synthetic_mean_scale, synthetic_cluster_std=synthetic_cluster_std,
        synthetic_n_classes=synthetic_n_classes,
        image_pool_side=image_pool_side,
        image_data_home=image_data_home,
    )
    n_samples, nviews = X_full.shape
    nclasses = int(Y_full.max()) + 1
    print(f"{dataset_name}: {n_samples} samples, {nviews} views, {nclasses} classes "
          f"(free: '{feature_names[0]}', {nviews - 1} paid)")

    raw_feature_costs = np.array(
        generate_modality_costs_heterogeneous(n_features=nviews, dataset_name=dataset_name),
        dtype=np.float64,
    )
    costs = raw_feature_costs / raw_feature_costs.sum()
    paid_costs = costs[1:]
    print(f"feature_costs[0] (free) = {costs[0]}, paid costs (normalized, sum(costs)==1): "
          f"min={paid_costs.min():.4f}, max={paid_costs.max():.4f}, mean={paid_costs.mean():.4f}, "
          f"sum={paid_costs.sum():.4f}")
    print(f"Stage-2 acquisition: greedy, "
          f"center_update={center_update!r}, alpha_ucb={alpha_ucb:g}"
          + (f", reward_update={reward_update!r}" if center_update == "reward_estimates" else ""))

    all_results = []

    for seed in seeds:
        seed_start = time.time()
        print(f"\n{'=' * 60}\n=== SEED {seed} ({dataset_name}, {nclasses} classes) ===\n{'=' * 60}")

        train_idx, test_idx = split_train_inference(n_samples, seed=seed)
        X_train, Y_train = X_full[train_idx], Y_full[train_idx]
        X_test, Y_test = X_full[test_idx], Y_full[test_idx]
        n_train, n_test = len(train_idx), len(test_idx)
        n_total = n_train + n_test

        rng = np.random.default_rng(seed=seed)

        for budget_fraction in budget_fractions:
            total_budget = budget_fraction * n_total
            train_inference_split = n_train / n_total
            training_budget = train_inference_split * total_budget
            inference_budget = total_budget - training_budget

            init_fractions = np.linspace(0, budget_fraction, n_init_fraction_points).tolist()

            for init_fraction in init_fractions:
                if init_fraction >= budget_fraction:
                    continue

                n_init_samples = int(n_train * init_fraction)
                train_start = time.time()
                centers, init_error = initialize_centers_multiclass(
                    x=X_train, y=Y_train, n_init_samples=n_init_samples,
                    k_clusters=nclasses, m_modalities=nviews, rng=rng,
                    pred_rule=pred_rule,
                )
                stage2_result = run_alg_greedy_multiclass(
                    x=X_train, y=Y_train, centers=centers, costs=costs,
                    T1=n_init_samples, training_budget=training_budget, rng=rng,
                    center_update=center_update, reward_update=reward_update,
                    alpha_ucb=alpha_ucb, lr=lr,
                    step_size=step_size, lambda_max=lambda_max,
                    pred_rule=pred_rule,
                )
                train_time = time.time() - train_start
                two_stage_error = calculate_two_stage_error(
                    T=n_train, T1=n_init_samples,
                    stage2_error=stage2_result['error_rate'], init_err=init_error,
                )

                # Inference classifies with whatever Stage 2 left behind:
                # the untouched Stage-1 centres under frozen/counts, the
                # updated ones under full/bandit. See module docstring (4).
                inference_centers = stage2_result['centers']

                if run_inference:
                    inference_start = time.time()
                    inference_result = run_inference_lp_dataset_colgen_multiclass(
                        X_inference=X_test, Y_inference=Y_test,
                        learned_centers=inference_centers, costs=costs,
                        inference_budget=inference_budget, rng=rng,
                        pred_rule=pred_rule,
                    )
                    inference_time = time.time() - inference_start

                    total_reward = (
                        (1 - init_error) * n_init_samples
                        + stage2_result['avg_reward'] * stage2_result['T2']
                        + inference_result['inference_accuracy'] * n_test
                    ) / max(1, n_train + n_test)
                else:
                    inference_result = {
                        'inference_accuracy': float('nan'),
                        'inference_error': float('nan'),
                        'inference_f1': float('nan'),
                        'inference_auroc': float('nan'),
                        'actual_cost': 0.0,
                        'num_masks_inference': np.nan,
                    }
                    inference_time = 0.0
                    total_reward = (
                        (1 - init_error) * n_init_samples
                        + stage2_result['avg_reward'] * stage2_result['T2']
                    ) / max(1, n_train)

                all_results.append({
                    'seed': seed,
                    'warm_start': False,          # no expert weights here
                    'nclasses': nclasses,
                    'budget_fraction': budget_fraction,
                    'init_fraction': init_fraction,
                    'Experts': np.nan,            # greedy enumerates nothing
                    'center_update': center_update,
                    'reward_update': reward_update if center_update == "reward_estimates" else "",
                    'alpha_ucb': alpha_ucb,
                    'avg_views_acquired': stage2_result['avg_views_acquired'],
                    'n_unique_masks': stage2_result['n_unique_masks'],
                    'n_train': n_train,
                    'n_test': n_test,
                    'T1': n_init_samples,
                    'T2': stage2_result['T2'],
                    'total_budget': total_budget,
                    'training_budget': training_budget,
                    'initialization_budget': n_init_samples * 1.0,
                    'training_budget_spent': stage2_result['training_budget_spent'],
                    'training_remaining_budget': stage2_result['training_remaining_budget'],
                    'lambda_final': stage2_result['lambda_final'],
                    'train_time_sec': train_time,
                    'inference_time_sec': inference_time,
                    'error_rate': stage2_result['error_rate'],
                    'avg_reward': stage2_result['avg_reward'],
                    'avg_lagrangian_reward': stage2_result['avg_lagrangian_reward'],
                    'total_reward': total_reward,
                    'train_f1': stage2_result['train_f1'],
                    'train_auroc': stage2_result['train_auroc'],
                    'init_error': init_error,
                    'two_stage_error': two_stage_error,
                    'inference_length': n_test,
                    'inference_budget': inference_budget,
                    'inference_accuracy': inference_result['inference_accuracy'],
                    'inference_error': inference_result['inference_error'],
                    'inference_f1': inference_result['inference_f1'],
                    'inference_auroc': inference_result['inference_auroc'],
                    'inference_actual_cost': inference_result['actual_cost'],
                    'num_masks_inference': inference_result['num_masks_inference'],
                })

        seed_elapsed = time.time() - seed_start
        print(f"\n  [SEED {seed}] wall-clock time: {seed_elapsed:.1f}s "
              f"({len(budget_fractions)} budget fractions x up to "
              f"{n_init_fraction_points} init fractions)")
        for row in all_results:
            if row['seed'] == seed and 'seed_time_sec' not in row:
                row['seed_time_sec'] = seed_elapsed

    return all_results


def save_results_to_excel(all_results, dataset_name, filename=None):
    """Same shape as two_stage_multiclass_runner.save_results_to_excel.
    'center_update' is a string column, so it is excluded from the numeric
    aggregation but kept in the Detailed sheet."""
    if filename is None:
        filename = f"results_two_stage_greedy_{dataset_name}.xlsx"

    df = pd.DataFrame(all_results)
    detailed_df = df.copy()

    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns.tolist()
    non_avg_cols = ['seed']
    avg_cols = [c for c in numeric_cols
                if c not in non_avg_cols and c not in ('budget_fraction', 'init_fraction')]

    averaged_df = (
        df.groupby(['budget_fraction', 'init_fraction'])[avg_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    averaged_df.columns = [
        col[0] if col[1] == '' else f'{col[0]}_{col[1]}'
        for col in averaged_df.columns
    ]

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        averaged_df.to_excel(writer, sheet_name='Averaged', index=False)

    wb = load_workbook(filename)
    _style_sheet(wb['Detailed'])
    _style_sheet(wb['Averaged'])
    wb.save(filename)
    print(f'Results saved to {filename}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run two_stage_multiclass_greedy.py (two_stage with a submodular-greedy "
                    "Stage-2 acquisition policy instead of EXP4) on a real AFA-Benchmark "
                    "dataset or the synthetic dataset."
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-modalities", type=str, default="all",
                        help="Integer, or 'all' (DEFAULT) for no truncation. Unlike "
                             "two_stage_multiclass_runner.py there is no 2^(nviews-1) expert "
                             "enumeration here, so this no longer needs to be kept small.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--n-init-fraction-points", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES)
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS)
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                        help="synthetic only: how many classes to generate.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED)
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE)
    parser.add_argument("--output-xlsx", type=str, default=None)
    parser.add_argument("--center-update", choices=CENTER_UPDATE_MODES, default="reward_full",
                        help="What Stage 2 is allowed to update. 'frozen': nothing (fully "
                             "static oracle). 'counts' (DEFAULT): visitation counts only, so "
                             "the optimism bonus decays for acquired views while the "
                             "classifier stays exactly the Stage-1 one -- the apples-to-apples "
                             "comparison against two_stage's EXP4. 'full'/'bandit': also "
                             "update the centres, using gmm_multiclass_submodular's "
                             "full/bandit-feedback rules. NOTE: 'full' reveals y_true every "
                             "round, which EXP4 never sees -- see run_alg_greedy_multiclass's "
                             "comparability warning.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                        help="--center-update reward_estimates ONLY. 'subsets' (DEFAULT): "
                             "counterfactual replay -- the acquired row was paid for on every "
                             "selected view, so every SUB-combination of it is scored for "
                             "free, giving 2^(|S|-1) observations per round instead of 1. "
                             "Needs y_true, so it is FULL feedback. 'selected': update only "
                             "the played combination from its 0/1 reward (bandit feedback); "
                             "with 2^(nviews-1) arms this barely moves off the Stage-1 prior "
                             "and is best treated as the ablation showing why replay matters.")
    parser.add_argument("--alpha-ucb", type=float, default=2.0,
                        help="Optimism scale in the greedy oracle's exploration bonus "
                             "sqrt(alpha_ucb*log(t+1))/sqrt(count) (default 2.0, same as "
                             "submodular). 0 disables optimism entirely -- combined with "
                             "--center-update frozen that makes the oracle deterministic "
                             "given lambda.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Complementary-label gradient step, --center-update bandit only "
                             "(default 1e-2, same as submodular).")
    parser.add_argument("--step-size", type=float, default=1.0,
                        help="Stage-2 OMD dual (lambda) ascent step size (default 1.0).")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                        help="Stage-2 OMD dual variable clip ceiling (default 10.0).")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Run ONLY Stage 1 + Stage-2 training; skip the Stage-2 LP "
                             "inference step. Same semantics as the EXP4 runner's flag.")
    parser.add_argument("--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE,
                        help="mnist/fashion_mnist only: block-average each 28x28 image down "
                             "to x*x features. Can be larger here than for two_stage (no "
                             "expert enumeration), but the Stage-2 LP pricing still bounds it.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                        help="mnist/fashion_mnist only: fetch_openml cache directory. See "
                             "core.datasets.DEFAULT_OPENML_DATA_HOME.")
    parser.add_argument("--pred-rule", choices=PRED_RULES, default="nearest_center",
                        help="Hard-decision rule, shared with two_stage. The two options "
                             "are mathematically equivalent -- see two_stage_multiclass.py.")
    args = parser.parse_args()

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    t0 = time.time()
    all_results = run_experiment(
        args.dataset,
        max_modalities=max_modalities,
        seeds=seeds,
        budget_fractions=budget_fractions,
        n_init_fraction_points=args.n_init_fraction_points,
        data_path=args.data_path,
        max_samples=args.max_samples,
        synthetic_n_samples=args.n_samples,
        synthetic_n_views=args.n_views,
        synthetic_seed=args.synthetic_seed,
        synthetic_mean_scale=args.mean_scale,
        synthetic_n_classes=args.num_classes,
        center_update=args.center_update,
        reward_update=args.reward_update,
        alpha_ucb=args.alpha_ucb,
        lr=args.lr,
        step_size=args.step_size,
        lambda_max=args.lambda_max,
        run_inference=not args.skip_inference,
        image_pool_side=args.image_pool_side,
        image_data_home=args.image_cache_dir,
        pred_rule=args.pred_rule,
    )

    df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}\nSUMMARY (mean +/- std across seeds) -- {args.dataset}\n{'=' * 70}")
    print(f"{'Budget':<10}{'InitFrac':<12}{'ErrorRate':>14}{'TwoStageErr':>16}"
          f"{'InfErr':>12}{'InfSpent':>12}{'Views':>9}")
    print("-" * 85)
    for (frac, init_f), sub in df.groupby(['budget_fraction', 'init_fraction']):
        print(
            f"{frac:<10.2f}{init_f:<12.3f}"
            f"{sub['error_rate'].mean():>10.3f}+/-{sub['error_rate'].std():.3f}"
            f"{sub['two_stage_error'].mean():>12.3f}+/-{sub['two_stage_error'].std():.3f}"
            f"{sub['inference_error'].mean():>8.3f}"
            f"{sub['inference_actual_cost'].mean():>12.4f}"
            f"{sub['avg_views_acquired'].mean():>9.2f}"
        )

    print(f"\nPer-seed wall-clock time (s): "
          f"{df.groupby('seed')['seed_time_sec'].first().to_dict()}")

    # center_update MUST be in the tag, or two runs differing only in mode
    # write the SAME auto-filename and silently clobber each other.
    cu_tag = f"_{args.center_update}"
    if args.center_update == "reward_estimates":
        cu_tag += f"-{args.reward_update}"
    ti_tag = "_trainonly" if args.skip_inference else ""
    maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
    pr_tag = "_pairwisevote" if args.pred_rule == "pairwise_vote" else ""
    au_tag = f"_alpha{args.alpha_ucb:g}" if args.alpha_ucb != 2.0 else ""
    classes_tag = ""
    if args.dataset in SYNTHETIC_DATASETS:
        n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
        classes_tag = f"_classes{n_classes}"
    save_results_to_excel(
        all_results, args.dataset,
        filename=args.output_xlsx or (
            f"results_two_stage_greedy_{args.dataset}_max{maxmod_label}"
            f"_seeds{len(seeds)}{cu_tag}{ti_tag}{pr_tag}{au_tag}{classes_tag}.xlsx"
        ),
    )
    print(f"Execution time: {time.time() - t0:.1f} seconds")