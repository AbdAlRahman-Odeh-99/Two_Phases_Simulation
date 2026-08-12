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
3. New: --acquisition {greedy,lp_chain,lp_full}, --reward-update,
   --alpha-ucb. There is deliberately NO warmup flag: Stage 1 IS the
   warmup, and a second one measured harmful.
4. Stage-2 INFERENCE uses the centres run_alg_greedy_multiclass RETURNS.
   Every acquisition mode holds the classifier FROZEN at its Stage-1
   values, so those are the Stage-1 centres by value -- reading them back
   off the result dict is a no-op today, kept because it is the correct
   contract if a learning mode is ever reintroduced.
5. Extra columns: 'acquisition', 'avg_views_acquired', 'n_unique_masks',
   'Experts' is emitted as NaN to keep the column
   present and the two runners' frames concatenable.

=== Column rename (breaking) ===
The old 'center_update' and 'action_space' columns are GONE, replaced by
a single 'acquisition' column. Workbooks written before this change do
not concatenate directly; map center_update="frozen" -> "greedy" and
center_update="reward_estimates" + action_space=X -> "lp_"+X, and drop
rows from the four deleted centre-learning modes, which have no
equivalent.

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
from core.excel_utils import _style_sheet, serialize_selected_subsets

from core.multiclass_common import (
    PRED_RULES,
    initialize_centers_multiclass,
    run_inference_lp_dataset_colgen_multiclass,
)
from core.optimal_static import synthetic_true_means
from core.two_stage_utils import calculate_two_stage_error
from two_stage_greedy.two_stage_multiclass_greedy import (
    ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
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
    acquisition="greedy",
    reward_update="subsets",
    alpha_ucb=2.0,
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
    for acquisition / reward_update / alpha_ucb semantics.

    run_inference: if False, Stage-2 LP inference is skipped, all
        inference_* columns become NaN and Total Reward becomes the
        train-only sample-weighted accuracy -- exactly as in the EXP4
        runner, so --skip-inference rows remain comparable BETWEEN the two
        methods (but not against a full run of either).
    """
    if dataset_name not in ALL_DATASETS:
        msg = f"dataset_name must be one of {ALL_DATASETS}, got {dataset_name!r}"
        raise ValueError(msg)
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                         f"got {acquisition!r}")

    if acquisition in ORACLE_ACQUISITION_MODES and dataset_name not in SYNTHETIC_DATASETS:
        # Hoisted ABOVE load_dataset_as_numpy deliberately. The true-means
        # recovery below has to wait for nviews, but this check does not, and
        # a real-dataset load can be slow or can fail on its own (network
        # fetch, missing cache) -- so the user would eat that wait, or a
        # confusing ConnectionError, before reaching a rejection that was
        # decidable from the flags alone.
        raise ValueError(
            f"acquisition={acquisition!r} needs the TRUE generative means, which "
            f"exist only for the synthetic datasets {SYNTHETIC_DATASETS}; got "
            f"{dataset_name!r}. Substituting a sample mean would turn the oracle "
            f"into a plug-in estimate the learning modes can beat, which is worse "
            f"than no ceiling at all. Use acquisition='lp_full' on real data.")

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
    _has_ru = acquisition not in ("greedy",) + ORACLE_ACQUISITION_MODES
    print(f"Stage-2 acquisition={acquisition!r}, alpha_ucb={alpha_ucb:g}"
          + (f", reward_update={reward_update!r}" if _has_ru else ""))

    # ── ORACLE ACQUISITION SETUP (acquisition="lp_full_opt" only) ──
    # Recovered ONCE per run: the generative means belong to the DATASET,
    # not to a seed's split or a budget/init fraction, so every cell of the
    # sweep shares them.
    #
    # n_views_used=nviews, NOT synthetic_n_views. load_dataset_as_numpy
    # applies features[:, :max_modalities] to synthetic data too, so the
    # generator's width and the width in X_full diverge whenever
    # --max-modalities is set. synthetic_true_means has to draw at the FULL
    # width and truncate after (numpy fills row-major), and nviews is what
    # tells it where to cut. Passing synthetic_n_views as the used width
    # instead yields plausible-looking but wrong means -- an oracle that has
    # quietly stopped being one, with nothing visibly failing.
    true_means = None
    if acquisition in ORACLE_ACQUISITION_MODES:
        # Dataset eligibility was already rejected above; this is the recovery,
        # which needs nviews and so has to happen after the load.
        true_means = synthetic_true_means(
            dataset_name,
            synthetic_n_views=synthetic_n_views,
            n_views_used=nviews,
            synthetic_seed=synthetic_seed,
            mean_scale=synthetic_mean_scale,
            n_classes=synthetic_n_classes,
        )
        print(f"  [oracle] recovered true means for acquisition={acquisition}: "
              f"shape {true_means.shape}")

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

            if budget_fraction == 0:
                # Produce exactly one zero-budget baseline row.
                init_fractions = [0.0]
            else:
                init_fractions = np.linspace(
                    0, budget_fraction, n_init_fraction_points
                ).tolist()

            for init_fraction in init_fractions:
                if budget_fraction > 0 and init_fraction >= budget_fraction:
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
                    acquisition=acquisition, reward_update=reward_update,
                    true_means=true_means,
                    alpha_ucb=alpha_ucb,
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
                    'acquisition': acquisition,
                    'reward_update': reward_update if _has_ru else "",
                    'n_arms': stage2_result.get('n_arms', 0),
                    'alpha_ucb': alpha_ucb,
                    'avg_views_acquired': stage2_result['avg_views_acquired'],
                    'n_unique_masks': stage2_result['n_unique_masks'],
                    'Selected Subsets': serialize_selected_subsets(
                        stage2_result['selected_subsets']),
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
    'acquisition' is a string column, so it is excluded from the numeric
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
    parser.add_argument("--acquisition", choices=ACQUISITION_MODES, default="greedy",
                        help="The per-round ACQUISITION policy, and the one axis on which "
                             "this method and --method submodular are directly comparable. "
                             "'greedy' (DEFAULT): the submodular greedy oracle, re-derived "
                             "each round from the frozen Stage-1 centres. 'lp_chain': UCB "
                             "reward estimates over the nviews+1 nested greedy chain with "
                             "the per-round budgeted LP solved exactly; runs at any nviews. "
                             "'lp_full': the same over the FULL 2^(nviews-1) enumeration -- "
                             "the small-nviews fidelity check for what the chain restriction "
                             "costs, capped at MAX_REWARD_ESTIMATE_VIEWS views. All three "
                             "hold the classifier frozen after Stage 1.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                        help="--acquisition lp_chain / lp_full ONLY. 'subsets' (DEFAULT): "
                             "counterfactual replay -- the acquired row was paid for on every "
                             "selected view, so every SUB-combination of it is scored for "
                             "free, giving 2^(|S|-1) observations per round instead of 1. "
                             "Needs y_true, so it is FULL feedback. 'selected': update only "
                             "the played combination from its 0/1 reward (bandit feedback); "
                             "with the full action space this barely moves off the Stage-1 "
                             "prior and is best treated as the ablation showing why replay "
                             "matters. Inert under --acquisition greedy.")
    parser.add_argument("--alpha-ucb", type=float, default=2.0,
                        help="Optimism scale in the exploration bonus "
                             "sqrt(alpha_ucb*log(t+1))/sqrt(count) (default 2.0, same as "
                             "submodular). 0 disables optimism entirely -- combined with "
                             "--acquisition greedy that makes the oracle deterministic "
                             "given lambda.")
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
        acquisition=args.acquisition,
        reward_update=args.reward_update,
        alpha_ucb=args.alpha_ucb,
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

    # acquisition MUST be in the tag, or two runs differing only in policy
    # write the SAME auto-filename and silently clobber each other.
    cu_tag = f"_{args.acquisition}"
    # No reward_update suffix for greedy (no arms) or the oracle modes (exact
    # arm values, never scored) -- it would name a scoring scope that never ran.
    if args.acquisition not in ("greedy",) + ORACLE_ACQUISITION_MODES:
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