"""
Driver for two_stage_asymmetric.py.

Mirrors gmm_bandit_runner.py's structure exactly, so the two
algorithms (two_stage's two-phase EXP4/Hedge-BwK bandit vs. gmm_2class's
OMD-dual bandit) can be compared on IDENTICAL data, splits, and costs.

This file does NOT modify two_stage_asymmetric.py -- it imports its
functions (generate_view_combinations, generate_combination_costs_heterogeneous,
initialize_centers, run_alg, calculate_two_phase_error, solve_LP,
predict_single_combination) UNCHANGED, and only replaces:
  1. generate_synthetic_data() (make_blobs) -> real/synthetic dataset
     loading, reusing core.datasets.load_dataset_as_numpy (the SAME shared
     loader gmm_bandit_runner.py and gmm_submodular_runner.py use) so all
     three runners load datasets identically.
  2. main()'s driver loop -> adapted for a single dataset, sweeping
     seeds x budget_fractions x init_fractions (same two-level sweep as the
     synthetic script), instead of repeated synthetic trials.
  3. run_inference_lp()'s label matching -> see label-matching note below.

=== Label matching note ===
The synthetic script's run_inference_lp() calls
utils.match_cluster_labels(learned_centers, true_means) to Hungarian-match
learned cluster centers to the TRUE generative means from make_blobs. Real
data has no such ground-truth generative means. However, this re-matching
step is not actually needed here: initialize_centers() seeds
learned_centers[yi] = xi the very first time class yi is observed, so
learned center row k already IS the running estimate of class k's mean by
construction -- centers are supervised-initialized, not an unsupervised/
arbitrary cluster order the way gmm_2class_bandit_asymmetric's est_means
initialization is (which starts as 2 random real data rows with no label
information, so it needs its own post-hoc Hungarian match against true
labels). So this driver uses the identity mapping {0: 0, 1: 1, ...}
instead of match_cluster_labels.

=== Costs ===
Same normalized-to-sum-1 cost convention as the (now-updated)
gmm_bandit_runner.py: draw the heterogeneous per-feature cost
SHAPE from datasets.generate_modality_costs_heterogeneous (fixed per
dataset, not redrawn per seed), then rescale the whole vector so
sum(costs) == 1 -- identical in spirit to two_stage_asymmetric.py's own
generate_per_view_costs, and to main_emp/gmm's synthetic cost block.
With sum(costs) == 1, total_budget = budget_fraction * n_total, exactly
matching the synthetic script's convention (no extra scaling factor).

=== Phase 2 now uses column generation (0.5-factor reward convention) ===
run_inference_lp_dataset_colgen (added here, mirrors
two_stage_asymmetric.run_inference_lp_colgen) solves the SAME LP that
generate_view_combinations(nviews) + solve_LP used to solve, via
core.lp_colgen.solve_lp_policy_colgen, without ever building the full
2^(nviews-1) combo list for Phase 2. Uses `lp_reward`'s 0.5-factor
formula (matching gain_func/optimal_branch_and_bound), NOT the no-0.5
formula the exhaustive run_inference_lp_dataset's own locally-computed
combo_reward_estimates uses -- see core/lp_colgen.py's docstring for the
discrepancy. The old exhaustive
run_inference_lp_dataset is still importable for small-nviews
comparison/debugging, just no longer wired into run_experiment's default
path.

=== Scalability caveat (Phase 1 -- NOT fixed by the above) ===
generate_view_combinations() enumerates 2^(nviews-1) combos (every subset
of the paid features, with the free view forced into every combo), and
run_alg's EXP4/Hedge-BwK bandit needs ALL of them as "experts" --
`weights = np.ones(n_experts)` has to exist for the full combo list before
training even starts, regardless of what Phase 2 does. Colgen only
removes Phase 2's LP wall; Phase 1's expert-count requirement is a
different, structural constraint (the whole point of the EXP4 regret
bound is defined over the full expert set) that isn't fixable the same
way. Keep nviews <= MAX_RECOMMENDED_MODALITIES or this becomes
intractable -- use --max-modalities to truncate real datasets with many
more paid features (ckd=24, actg175=23, bank_marketing=17, miniboone=50,
physionet=41).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score

from core.datasets import (
    ALL_DATASETS,
    MAX_RECOMMENDED_MODALITIES,
    SYNTHETIC_CLUSTER_STD,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_SAMPLES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    generate_modality_costs_heterogeneous,
    load_dataset_as_numpy,
    split_train_inference,
)
from core.excel_utils import _style_sheet
from core.lp_colgen import solve_lp_policy_colgen, sample_lp_colgen_policy

from two_stage.two_stage_asymmetric import (
    generate_view_combinations,
    generate_combination_costs_heterogeneous,
    initialize_centers,
    run_alg,
    calculate_two_phase_error,
    solve_LP,
    predict_single_combination,
)


def run_inference_lp_dataset(
    X_inference, Y_inference,
    learned_centers,
    view_combinations, combo_costs,
    inference_budget, m_modalities,
    rng,
    label_map=None, noise_var=1.0,
):
    """Real-data counterpart to two_stage_asymmetric.run_inference_lp -- same
    LP-solve + physical-sampling logic, but skips match_cluster_labels (no
    ground-truth generative means available). See module docstring for why
    the identity label_map is the correct substitute here, not a workaround.

    NOT used by run_experiment by default anymore -- see
    run_inference_lp_dataset_colgen, which solves the identical LP via
    column generation instead of the full view_combinations power set, and
    is what run_experiment calls now for Phase 2. Kept here for
    small-nviews comparison/debugging.

    combo_reward_estimates is now computed HERE, locally, from
    `learned_centers` (no-0.5 `norm.sf(sqrt(snr))` formula, same as
    two_stage_asymmetric.run_inference_lp) -- run_alg no longer builds it
    (nothing on the colgen path needs it), so this function computes its
    own copy instead of expecting it passed in from ph2_result.
    """
    n = len(X_inference)
    if label_map is None:
        label_map = {k: k for k in range(learned_centers.shape[0])}

    n_views = learned_centers.shape[1]
    mean_diff_sq = np.square(0.5 * (learned_centers[0, :] - learned_centers[1, :]) / np.sqrt(noise_var))
    combo_reward_estimates = {}
    for c in view_combinations:
        mask = np.zeros(n_views, dtype=bool)
        mask[np.array(c) - 1] = True
        snr = np.sum(mean_diff_sq[mask])
        err_rate = norm.sf(np.sqrt(snr))
        combo_reward_estimates[c] = 1 - err_rate

    lp_weights = solve_LP(
        view_combinations=view_combinations,
        combinations_rewards_estimates=combo_reward_estimates,
        combination_costs=combo_costs,
        M_MODALITIES=m_modalities,
        Budget=inference_budget,
        T=n,
    )

    combos_list = list(view_combinations)
    probabilities = np.array([lp_weights[combo] for combo in combos_list])
    probabilities = np.clip(probabilities, 0, 1)
    probabilities /= np.sum(probabilities)

    actual_inference_rewards = []
    actual_inference_cost = 0.0
    remaining_inference_budget = inference_budget

    # Same sign convention as two_stage_asymmetric.run_alg's AUROC
    # computation / run_inference_lp's. label_map is identity here (see
    # module docstring), so sign_factor is always +1 in practice, but this
    # keeps the convention identical if that ever changes.
    sign_factor = 1.0 if label_map.get(0, 0) == 0 else -1.0
    y_pred_mapped = np.zeros(n, dtype=int)
    evidence_for_class1 = np.zeros(n)

    for i in range(n):
        sampled_idx = rng.choice(len(combos_list), p=probabilities)
        sampled_combo = combos_list[sampled_idx]

        cost_of_combo = combo_costs[sampled_combo]
        if remaining_inference_budget - cost_of_combo < 0:
            sampled_combo = (1,)  # fallback: the free view alone
            cost_of_combo = combo_costs[sampled_combo]
        else:
            remaining_inference_budget -= cost_of_combo
            actual_inference_cost += cost_of_combo

        raw_pred, score = predict_single_combination(X_inference[i], learned_centers, sampled_combo, return_score=True)
        matched_pred = label_map.get(raw_pred, raw_pred)
        actual_inference_rewards.append(int(matched_pred == Y_inference[i]))
        y_pred_mapped[i] = matched_pred
        evidence_for_class1[i] = sign_factor * score

    inference_accuracy = float(np.mean(actual_inference_rewards))

    inference_f1 = f1_score(Y_inference, y_pred_mapped, zero_division=0)
    try:
        inference_auroc = roc_auc_score(Y_inference, evidence_for_class1)
    except ValueError:
        inference_auroc = float("nan")

    return {
        "lp_weights": lp_weights,
        "inference_accuracy": inference_accuracy,
        "inference_error": float(1 - inference_accuracy),
        "inference_f1": inference_f1,
        "inference_auroc": inference_auroc,
        "actual_cost": actual_inference_cost,
    }


def run_inference_lp_dataset_colgen(
    X_inference, Y_inference,
    learned_centers, costs,
    inference_budget, m_modalities,
    rng, label_map=None, noise_var=1.0,
):
    """Column-generation counterpart to run_inference_lp_dataset -- same
    role as two_stage_asymmetric.run_inference_lp_colgen (0.5-factor
    reward convention, see core/lp_colgen.py), adapted for real data the
    same way run_inference_lp_dataset adapts run_inference_lp (identity
    label_map instead of match_cluster_labels -- see module docstring's
    Label matching note).

    Thin wrapper: builds the identity label_map (the one thing genuinely
    specific to real data with no ground-truth generative means), then
    delegates to core.lp_colgen.sample_lp_colgen_policy, shared with
    two_stage_asymmetric.run_inference_lp_colgen's synthetic-data
    counterpart.

    Takes `costs` (the plain per-view array) instead of
    `view_combinations`/`combo_costs`/`combo_reward_estimates` -- doesn't
    build any 2^(nviews-1) structure, so this scales to physionet (41) /
    miniboone (50) for Phase 2 regardless of --max-modalities. Phase 1
    (run_alg) still needs the full view_combinations list built in
    run_experiment above regardless (EXP4 requires one weight per expert)
    -- this only replaces Phase 2's LP call site.
    """
    if label_map is None:
        label_map = {k: k for k in range(learned_centers.shape[0])}
    return sample_lp_colgen_policy(
        X_inference=X_inference, Y_inference=Y_inference,
        learned_centers=learned_centers, costs=costs,
        inference_budget=inference_budget, rng=rng, label_map=label_map,
        predict_fn=predict_single_combination, noise_var=noise_var,
    )


def run_experiment(
    dataset_name,
    max_modalities=10,
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
):
    """
    Counterpart to two_stage_asymmetric.main(), reusing ALL of its Phase-1
    (init + EXP4/Hedge-BwK) / Phase-2 (LP inference) functions unchanged.
    Sweeps seeds x budget_fractions x init_fractions, exactly the two-level
    sweep the synthetic main() does per trial. dataset_name may be any of
    the 5 real AFA-Benchmark tabular datasets OR "synthetic_asymmetric" /
    "synthetic_symmetric" -- see datasets.load_binary_afa_dataset.

    ALIGNED with gmm_bandit_runner.run_experiment for
    direct comparability: same load_dataset_as_numpy, same
    split_train_inference(seed=seed) 80/20 partitions (no val), same
    normalized (sum(costs)==1) cost convention, same total_budget =
    budget_fraction * n_total formula.

    max_samples: caps a REAL dataset to at most this many rows before
        splitting. Ignored for synthetic datasets; use synthetic_n_samples
        for those instead.
    """
    if dataset_name not in ALL_DATASETS:
        msg = f"dataset_name must be one of {ALL_DATASETS}, got {dataset_name!r}"
        raise ValueError(msg)

    X_full, Y_full, feature_names = load_dataset_as_numpy(
        dataset_name, max_modalities=max_modalities, data_path=data_path,
        max_samples=max_samples, synthetic_n_samples=synthetic_n_samples,
        synthetic_n_views=synthetic_n_views, synthetic_seed=synthetic_seed,
        synthetic_mean_scale=synthetic_mean_scale, synthetic_cluster_std=synthetic_cluster_std,
    )
    n_samples, nviews = X_full.shape
    print(f"{dataset_name}: {n_samples} samples, {nviews} views "
          f"(free: '{feature_names[0]}', {nviews - 1} paid)")

    # Fixed per-dataset costs, normalized so sum(costs) == 1 -- see module
    # docstring. NOT redrawn per seed.
    raw_feature_costs = np.array(
        generate_modality_costs_heterogeneous(n_features=nviews, dataset_name=dataset_name),
        dtype=np.float64,
    )
    costs = raw_feature_costs / raw_feature_costs.sum()
    paid_costs = costs[1:]
    print(f"feature_costs[0] (free) = {costs[0]}, paid costs (normalized, sum(costs)==1): "
          f"min={paid_costs.min():.4f}, max={paid_costs.max():.4f}, mean={paid_costs.mean():.4f}, "
          f"sum={paid_costs.sum():.4f}")

    view_combinations = generate_view_combinations(nviews)
    n_experts = len(view_combinations)
    print(f"view_combinations (free view forced in every combo): {n_experts} experts")
    if nviews > MAX_RECOMMENDED_MODALITIES:
        print(f"WARNING: nviews={nviews} exceeds the recommended max of "
              f"{MAX_RECOMMENDED_MODALITIES} -- {n_experts} experts for run_alg's "
              f"weight vector AND solve_LP's linear program. This may be extremely "
              f"slow or effectively never finish -- see this module's docstring.")
    combo_costs = generate_combination_costs_heterogeneous(view_combinations, costs)

    all_results = []

    for seed in seeds:
        seed_start = time.time()
        print(f"\n{'=' * 60}\n=== SEED {seed} ({dataset_name}) ===\n{'=' * 60}")

        # 80/20 train/inference split (no val) -- see run_experiment's
        # docstring / split_train_inference's docstring.
        train_idx, test_idx = split_train_inference(n_samples, seed=seed)
        X_train, Y_train = X_full[train_idx], Y_full[train_idx]
        X_test, Y_test = X_full[test_idx], Y_full[test_idx]
        n_train, n_test = len(train_idx), len(test_idx)
        n_total = n_train + n_test

        rng = np.random.default_rng(seed=seed)

        for budget_fraction in budget_fractions:
            # sum(costs) == 1, so total_budget is directly frac * n_total,
            # matching the synthetic script's convention.
            total_budget = budget_fraction * n_total
            train_inference_split = n_train / n_total  # auto-derived
            training_budget = train_inference_split * total_budget
            inference_budget = total_budget - training_budget

            init_fractions = np.linspace(0, budget_fraction, n_init_fraction_points).tolist()

            for init_fraction in init_fractions:
                if init_fraction >= budget_fraction:
                    continue

                n_init_samples = int(n_train * init_fraction)
                train_start = time.time()
                centers, init_error = initialize_centers(
                    x=X_train, y=Y_train, n_init_samples=n_init_samples,
                    k_clusters=2, m_modalities=nviews, rng=rng,
                )
                ph2_result = run_alg(
                    x=X_train, y=Y_train, centers=centers,
                    view_combinations=view_combinations, combo_costs=combo_costs,
                    T1=n_init_samples, training_budget=training_budget, rng=rng,
                )
                train_time = time.time() - train_start
                two_phase_error = calculate_two_phase_error(
                    T=n_train, T1=n_init_samples,
                    ph2_error=ph2_result['error_rate'], init_err=init_error,
                )
                inference_start = time.time()
                inference_result = run_inference_lp_dataset_colgen(
                    X_inference=X_test, Y_inference=Y_test,
                    learned_centers=centers, costs=costs,
                    inference_budget=inference_budget, m_modalities=nviews, rng=rng,
                )
                inference_time = time.time() - inference_start

                # Comparable "Total Reward" -- SAME sample-weighted formula
                # bandit/submodular use ((n_train*train_reward +
                # n_inference*inference_reward) / n_total), in the SAME
                # 0/1-accuracy units, just split into the T1 init-phase
                # rounds (accuracy = 1 - init_error) and T2 bandit-training
                # rounds (accuracy = avg_reward) separately, since those are
                # the two quantities actually available here -- T1 + T2 ==
                # n_train always, so this is a sample-weighted average of
                # "was this round's prediction correct" over EVERY round of
                # the whole seed (init + bandit-training + inference), not
                # just a relabeled two_phase_error. Previously left as NaN
                # entirely; now filled in.
                total_reward = (
                    (1 - init_error) * n_init_samples
                    + ph2_result['avg_reward'] * ph2_result['T2']
                    + inference_result['inference_accuracy'] * n_test
                ) / max(1, n_train + n_test)

                all_results.append({
                    'seed': seed,
                    'budget_fraction': budget_fraction,
                    'init_fraction': init_fraction,
                    'Experts': n_experts,
                    'n_train': n_train,
                    'n_test': n_test,
                    'T1': n_init_samples,
                    'T2': ph2_result['T2'],
                    'total_budget': total_budget,
                    'training_budget': training_budget,
                    'initialization_budget': n_init_samples * 1.0,  # cost_full_modalities==1 convention
                    'training_budget_spent': ph2_result['training_budget_spent'],
                    'training_remaining_budget': ph2_result['training_remaining_budget'],
                    'train_time_sec': train_time,
                    'inference_time_sec': inference_time,
                    'error_rate': ph2_result['error_rate'],
                    'avg_reward': ph2_result['avg_reward'],
                    'avg_lagrangian_reward': ph2_result['avg_lagrangian_reward'],
                    'total_reward': total_reward,
                    'train_f1': ph2_result['train_f1'],
                    'train_auroc': ph2_result['train_auroc'],
                    'init_error': init_error,
                    'two_phase_error': two_phase_error,
                    'inference_length': n_test,
                    'inference_budget': inference_budget,
                    'inference_accuracy': inference_result['inference_accuracy'],
                    'inference_error': inference_result['inference_error'],
                    'inference_f1': inference_result['inference_f1'],
                    'inference_auroc': inference_result['inference_auroc'],
                    'inference_actual_cost': inference_result['actual_cost'],
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
    """Same shape as two_stage_asymmetric.save_results_to_excel, keyed by
    (seed, budget_fraction, init_fraction) instead of (trial, budget_fraction,
    init_fraction), plus the styled sheets used by the gmm real-data runner."""
    if filename is None:
        filename = f"results_two_stage_{dataset_name}.xlsx"

    df = pd.DataFrame(all_results)
    detailed_df = df.copy()

    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns.tolist()
    non_avg_cols = ['seed']
    avg_cols = [c for c in numeric_cols if c not in non_avg_cols and c not in ('budget_fraction', 'init_fraction')]

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
        description="Run two_stage_asymmetric.py's algorithm on a real AFA-Benchmark dataset."
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="ckd")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-modalities", type=str, default="10",
                         help=f"Integer, or 'all' for no truncation. Keep this <= "
                              f"{MAX_RECOMMENDED_MODALITIES} or generate_view_combinations's "
                              f"2^(nviews-1) enumeration (fed into run_alg's weight vector AND "
                              f"solve_LP's linear program) becomes intractable -- see module "
                              f"docstring. 'all' is provided for completeness but is NOT "
                              f"recommended for any of these 5 datasets.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap a REAL dataset to at most this many rows (reproducible "
                              "subsample; if the dataset already has fewer rows, all of them "
                              "are used). Ignored for synthetic datasets -- use --n-samples "
                              "instead.")
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--n-init-fraction-points", type=int, default=10,
                         help="Number of init_fraction (gamma) points swept per budget "
                              "fraction, linspace(0, budget_fraction, N) -- matches the "
                              "synthetic script's n_runs=10 default.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                         help="Comma-separated seeds. Each seed uses datasets."
                              "split_train_inference(seed=seed) for an 80/20 train/inference "
                              "partition.")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES,
                         help="synthetic_asymmetric/synthetic_symmetric only: how many rows to "
                              "generate. Ignored for real datasets.")
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS,
                         help="synthetic_asymmetric/synthetic_symmetric only: how many views to "
                              "generate. Ignored for real datasets.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED,
                         help="synthetic_asymmetric/synthetic_symmetric only: seed for the "
                              "generative means/make_blobs draw (independent of --seeds, which "
                              "controls the train/inference split).")
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE,
                         help="synthetic_asymmetric/synthetic_symmetric only: per-view means "
                              "drawn ~ Uniform(0, mean_scale).")
    parser.add_argument("--output-xlsx", type=str, default=None)
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
    )

    df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}\nSUMMARY (mean +/- std across seeds) -- {args.dataset}\n{'=' * 70}")
    print(f"{'Budget':<10}{'InitFrac':<12}{'ErrorRate':>14}{'TwoPhaseErr':>16}{'InfErr':>12}{'InfSpent':>12}")
    print("-" * 76)
    for (frac, init_f), sub in df.groupby(['budget_fraction', 'init_fraction']):
        print(
            f"{frac:<10.2f}{init_f:<12.3f}"
            f"{sub['error_rate'].mean():>10.3f}+/-{sub['error_rate'].std():.3f}"
            f"{sub['two_phase_error'].mean():>12.3f}+/-{sub['two_phase_error'].std():.3f}"
            f"{sub['inference_error'].mean():>8.3f}"
            f"{sub['inference_actual_cost'].mean():>12.4f}"
        )

    print(f"\nPer-seed wall-clock time (s): "
          f"{df.groupby('seed')['seed_time_sec'].first().to_dict()}")

    save_results_to_excel(all_results, args.dataset, filename=f"results_two_stage_{args.dataset}_max{max_modalities}_seeds{len(seeds)}.xlsx")
    print(f"Execution time: {time.time() - t0:.1f} seconds")