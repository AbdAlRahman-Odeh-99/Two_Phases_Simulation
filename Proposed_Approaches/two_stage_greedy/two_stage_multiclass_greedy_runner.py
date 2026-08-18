"""
two_stage_multiclass_greedy_runner.py

Driver for two_stage_multiclass_greedy.py.

=== Observability (see core/logging_utils.py) ===
Every print() became a logger call, so the same messages now carry a
timestamp and a level and go to BOTH the console and logs/{run_id}.log,
where the DEBUG detail the console suppresses is still recorded.

Four behavioural additions, none of which changes a computed number:

  * Each sweep cell runs inside `guard(...)`. A cell that raises is logged
    with its coordinates and recorded as a row with status="error"; the
    sweep continues. Previously one bad cell killed the job and discarded
    every completed seed.
  * Each finished row is appended to results/{run_id}.rows.jsonl the
    moment it exists, so a walltime kill leaves the completed work on disk.
  * Each cell's row carries the fine timing decomposition
    (core.logging_utils.TIMER_KEYS) ALONGSIDE the original train_time_sec /
    inference_time_sec / seed_time_sec, which are unchanged.
  * A Progress heartbeat reports done/total cells with an ETA.

The only schema changes are appended columns: the timing block, `status`
and `error_msg`. Existing column names, values and sheet names are
untouched, and the Averaged sheet now aggregates over status=="ok" rows
only, so an isolated failure cannot drag a mean toward NaN.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

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
from core.excel_utils import serialize_selected_subsets, style_and_save
from core.logging_utils import (
    Progress,
    cell_timers,
    get_logger,
    get_run,
    guard,
    setup_run,
    tick,
    timing_row,
)

from core.multiclass_common import (
    PRED_RULES,
    initialize_centers_multiclass,
    run_inference_lp_dataset_colgen_multiclass,
)
from core.optimal_static import synthetic_true_means
# Single definition of "does --reward-update do anything for these flags",
# replacing the two inline copies this file used to carry.
from core.submodular_greedy import (
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
)
from core.two_stage_utils import calculate_two_stage_error
from two_stage_greedy.two_stage_multiclass_greedy import (
    ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
    REWARD_ESTIMATES,
    REWARD_UPDATE_SCOPES,
    run_alg_greedy_multiclass,
)

log = get_logger("afa.two_stage.runner")


def init_fraction_grid(budget_fraction, n_points):
    """The init_fraction (gamma) points actually swept at one budget.

    EXACTLY the old inline logic -- `[0.0]` at budget_fraction 0, otherwise
    linspace(0, budget_fraction, n_points) with the points satisfying
    `init_fraction >= budget_fraction` dropped (i.e. the endpoint). Hoisted
    into a function for one reason: the progress heartbeat needs the cell
    COUNT before the sweep starts, and a second copy of this arithmetic
    that drifted from the loop's would make the ETA quietly wrong.
    """
    if budget_fraction == 0:
        # Exactly one zero-budget baseline row.
        return [0.0]
    return [f for f in np.linspace(0, budget_fraction, n_points).tolist()
            if f < budget_fraction]


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
    reward_estimate="surrogate",
    reward_update="subsets",
    arm_elimination=False,
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
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, got {acquisition!r}")
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, got {reward_update!r}")
    # MEMBERSHIP check, matching gmm_multiclass_submodular_runner. Without
    # it a misspelling ("unbaised") does not raise anywhere: every test
    # downstream compares against the exact string "empirical", so the run
    # silently proceeds on the BIASED path while being reported -- and
    # filenamed -- as whatever was passed.
    if reward_estimate not in REWARD_ESTIMATES:
        raise ValueError(f"reward_estimate must be one of {REWARD_ESTIMATES}, got {reward_estimate!r}")
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

    run = get_run()
    trace_rounds = bool(run is not None and run.trace_rounds)

    with tick("t_data_load"):
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
    log.info("%s: %d samples, %d views, %d classes (free: '%s', %d paid)",
             dataset_name, n_samples, nviews, nclasses, feature_names[0], nviews - 1)

    raw_feature_costs = np.array(
        generate_modality_costs_heterogeneous(n_features=nviews, dataset_name=dataset_name),
        dtype=np.float64,
    )
    costs = raw_feature_costs / raw_feature_costs.sum()
    paid_costs = costs[1:]
    log.info("feature_costs[0] (free) = %s, paid costs (normalized, sum(costs)==1): "
             "min=%.4f, max=%.4f, mean=%.4f, sum=%.4f",
             costs[0], paid_costs.min(), paid_costs.max(), paid_costs.mean(),
             paid_costs.sum())

    uses_reward_update = _uses_empirical_arm_rewards(acquisition, reward_estimate)
    _has_ru = uses_reward_update
    log.info("Stage-2 acquisition=%r, reward_estimate=%r, alpha_ucb=%g%s",
             acquisition, reward_estimate, alpha_ucb,
             (f", reward_update={reward_update!r}" if _has_ru else ""))

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
        log.info("[oracle] recovered true means for acquisition=%s: shape %s",
                 acquisition, true_means.shape)

    all_results = []

    # Total cell count for the ETA, from the SAME grid function the loop
    # uses -- see init_fraction_grid's docstring.
    total_cells = len(seeds) * sum(
        len(init_fraction_grid(bf, n_init_fraction_points)) for bf in budget_fractions)
    progress = Progress(total_cells, label="cells", logger=log)
    log.info("sweep: %d seeds x %d budget fractions = %d cells",
             len(seeds), len(budget_fractions), total_cells)

    for seed in seeds:
        seed_start = time.time()
        log.info("=" * 60)
        log.info("=== SEED %s (%s, %d classes) ===", seed, dataset_name, nclasses)
        log.info("=" * 60)

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

            for init_fraction in init_fraction_grid(budget_fraction, n_init_fraction_points):
                cell = {"seed": int(seed), "budget_fraction": float(budget_fraction),
                        "init_fraction": float(init_fraction)}
                row = None

                # One bucket per cell. t_data_load is inherited so each row
                # is a complete account of its own time.
                with cell_timers(inherit=("t_data_load",)) as timers, \
                        guard(cell, logger=log) as outcome:
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
                        reward_estimate=reward_estimate,
                        arm_elimination=arm_elimination,
                        true_means=true_means,
                        alpha_ucb=alpha_ucb,
                        step_size=step_size, lambda_max=lambda_max,
                        pred_rule=pred_rule,
                        trace_rounds=trace_rounds,
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

                    row = {
                        'seed': seed,
                        'warm_start': False,          # no expert weights here
                        'nclasses': nclasses,
                        'budget_fraction': budget_fraction,
                        'init_fraction': init_fraction,
                        'Experts': np.nan,            # legacy column; this method does not use EXP4 experts
                        'acquisition': acquisition,
                        'reward_estimate': reward_estimate,
                        'reward_update': reward_update if _has_ru else "",
                        'n_arms': stage2_result.get('n_arms', 0),
                        'alpha_ucb': alpha_ucb,
                        'avg_views_acquired': stage2_result['avg_views_acquired'],
                        'n_unique_masks': stage2_result['n_unique_masks'],
                        'Selected Subsets': (
                            # With --trace-rounds the per-round detail lives in
                            # the trace sidecar, where it is actually usable;
                            # duplicating thousands of subsets into one Excel
                            # cell as well would just bloat the workbook.
                            "" if trace_rounds
                            else serialize_selected_subsets(stage2_result['selected_subsets'])),
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
                        'arm_elimination': stage2_result['arm_elimination'],
                        'initial_arms': stage2_result['initial_arms'],
                        'final_active_arms': stage2_result['final_active_arms'],
                        'num_eliminated': stage2_result['num_eliminated'],
                        'elimination_trace': str(stage2_result['elimination_trace']),
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
                        'n_budget_fallbacks_train': stage2_result.get('n_budget_fallbacks', np.nan),
                        'n_budget_fallbacks_inference': inference_result.get('n_budget_fallbacks', np.nan),

                    }

                    if trace_rounds and run is not None and stage2_result.get('round_trace'):
                        for rec in stage2_result['round_trace']:
                            run.emit_trace({**cell, **rec})

                    log.debug("cell %s -> two_stage_error=%.4f inference_error=%s "
                              "train=%.2fs inference=%.2fs",
                              cell, two_stage_error, inference_result['inference_error'],
                              train_time, inference_time)

                # -- outside the guard: record the cell either way ---------
                if row is None:
                    # The cell raised. Keep its coordinates and whatever the
                    # timers captured before the failure; every other column
                    # is simply absent, which pandas renders as NaN. Losing
                    # the rest of the sweep to this would be the expensive
                    # outcome, not losing this one row's metrics.
                    row = dict(cell)
                    row['nclasses'] = nclasses
                    row['acquisition'] = acquisition
                    row['reward_estimate'] = reward_estimate
                    row['n_train'] = n_train
                    row['n_test'] = n_test
                    row['training_budget'] = training_budget
                    row['inference_budget'] = inference_budget

                row['status'] = outcome.status
                row['error_msg'] = outcome.error
                row.update(timing_row(timers))

                all_results.append(row)
                if run is not None:
                    run.emit_row(row)
                progress.step(note=f"seed {seed} bf {budget_fraction:g}")

        seed_elapsed = time.time() - seed_start
        log.info("[SEED %s] wall-clock time: %.1fs (%d budget fractions x up to %d "
                 "init fractions)", seed, seed_elapsed, len(budget_fractions),
                 n_init_fraction_points)
        for row in all_results:
            if row['seed'] == seed and 'seed_time_sec' not in row:
                row['seed_time_sec'] = seed_elapsed

    n_failed = sum(1 for r in all_results if r.get('status') == 'error')
    if n_failed:
        log.warning("%d/%d cells FAILED and were recorded with status='error' -- see "
                    "the log for tracebacks and the manifest for the list",
                    n_failed, len(all_results))

    return all_results


def save_results_to_excel(all_results, dataset_name, filename=None, info_rows=None):
    """Same shape as two_stage_multiclass_runner.save_results_to_excel.
    'acquisition' is a string column, so it is excluded from the numeric
    aggregation but kept in the Detailed sheet.

    CHANGED in two ways, both additive:
      * the Averaged sheet aggregates over status=="ok" rows only, so one
        isolated failure cannot pull a cell's mean to NaN;
      * a third "Run Info" sheet carries the provenance manifest when a run
        context is active (info_rows=None -> no sheet, unchanged output).
    """
    if filename is None:
        filename = f"results_two_stage_{dataset_name}.xlsx"

    df = pd.DataFrame(all_results)
    detailed_df = df.copy()

    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df

    numeric_cols = ok_df.select_dtypes(include=[np.number]).columns.tolist()
    non_avg_cols = ['seed']
    avg_cols = [c for c in numeric_cols
                if c not in non_avg_cols and c not in ('budget_fraction', 'init_fraction')]

    averaged_df = (
        ok_df.groupby(['budget_fraction', 'init_fraction'])[avg_cols]
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

    style_and_save(filename, ['Detailed', 'Averaged'], info_rows=info_rows)
    log.info('Results saved to %s', filename)


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
                        help="Integer, or 'all'. Biased greedy itself does not enumerate subsets, "
                             "but reward_estimate='empirical' and lp_full require a full "
                             "2^(nviews-1) table and therefore need small nviews.")
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
                             "costs, capped at MAX_REWARD_ESTIMATE_VIEWS views. "
                             "'ucb_argmax' is the NOTEBOOK rule "
                             "(multiclass_supervised_unbiased_adaptive.ipynb): lp_full's "
                             "arm table and UCB estimates, but a DETERMINISTIC Lagrangian "
                             "argmax (argmax_S r_hat - lambda*cost + bonus) with greedy's "
                             "OMD dual in place of the LP, so it commits to one subset "
                             "instead of sampling a mixture. Same 2^(nviews-1) cap as "
                             "lp_full; --reward-estimate is inert under it. All of them "
                             "hold the classifier frozen after Stage 1.")
    parser.add_argument("--reward-estimate",
                        choices=list(REWARD_ESTIMATES),
                        default="surrogate",
                        help="What the per-arm reward estimates ARE. "
                             "'surrogate': the closed-form Bhattacharyya "
                             "proxy computed from the estimated means -- "
                             "needs no observations and no enumeration. "
                             "'empirical': a measured per-subset accuracy "
                             "table filled in by the containment replay; "
                             "needs 2^(nviews-1) arms, so nviews must stay "
                             "under MAX_REWARD_ESTIMATE_VIEWS. RENAMED from "
                             "biased/unbiased, which are NO LONGER ACCEPTED, "
                             "because 'unbiased' also names the "
                             "containment-replay acquisition (--acquisition "
                             "ucb_argmax, from sim_unbiased) and the two are "
                             "unrelated.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                        help="How empirical arm rewards are updated. "
                             "'subsets' replays every arm contained in the acquired subset; "
                             "'selected' updates only the played arm from its 0/1 reward. "
                             "Used by lp_chain/lp_full and by greedy when --reward-estimate "
                             "empirical. Ignored by surrogate greedy and lp_full_opt.")
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
    # ── observability flags (see core/logging_utils.py) ──
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="CONSOLE verbosity. The file log always keeps DEBUG, so "
                             "lowering this hides nothing permanently -- it only controls "
                             "what competes for attention in the PBS .o file.")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for {run_id}.log (default: logs/).")
    parser.add_argument("--trace-rounds", action="store_true",
                        help="Record one row PER STAGE-2 ROUND (subset, cost, lambda, "
                             "reward, remaining budget) to results/{run_id}.trace.jsonl, "
                             "and drop the 'Selected Subsets' Excel cell, whose content is "
                             "a strictly poorer version of the same thing. Off by default: "
                             "it is O(n_train) records per sweep cell.")
    parser.add_argument("--no-fine-timers", action="store_true",
                        help="Disable the fine-grained timing buckets (the t_* columns "
                             "become NaN). train/inference/seed timings are unaffected.")
    args = parser.parse_args()

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    uses_reward_update = _uses_empirical_arm_rewards(args.acquisition, args.reward_estimate)
    # Same scheme as run_proposed_methods.py and the submodular runner. The
    # previous version appended --reward-update TWICE for lp_chain/lp_full
    # (both the uses_reward_update test and the "not greedy" test fired),
    # producing names like "..._lp_chain-subsets-subsets_surrogate...".
    # The acquisition is now ALWAYS named, including the "greedy" default,
    # which surrogate greedy alone used to omit -- so a greedy run's filename
    # did not say "greedy" anywhere, and did not sort beside its own
    # lp_chain/lp_full siblings.
    # CHANGES default filenames for surrogate greedy runs only.
    cu_tag = f"_{args.acquisition}"
    if uses_reward_update:
        cu_tag += f"-{args.reward_update}"
    ti_tag = "_trainonly" if args.skip_inference else ""
    maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
    pr_tag = "_pairwisevote" if args.pred_rule == "pairwise_vote" else ""
    au_tag = f"_alpha{args.alpha_ucb:g}" if args.alpha_ucb != 2.0 else ""
    # ALWAYS tagged, including the "surrogate" default -- see the note in
    # gmm_multiclass_submodular_runner.py. This CHANGES default filenames.
    reward_estimate_is_live = (args.acquisition in ("greedy", "lp_chain"))
    re_tag = (f"_{args.reward_estimate}" if reward_estimate_is_live else "")
    dual_tag = f"_step{args.step_size:g}_lmax{args.lambda_max:g}"
    classes_tag = ""
    if args.dataset in SYNTHETIC_DATASETS:
        n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
        classes_tag = f"_classes{n_classes}"
    # MOVED ABOVE the run (it was computed after it). The filename is a pure
    # function of the arguments, so computing it first costs nothing and buys
    # two things: a bad --output-xlsx path fails in the first second instead
    # of after the sweep, and the run_id can be derived from it so the log,
    # the manifest, the JSONL checkpoint and the workbook all share a stem.
    output_xlsx = args.output_xlsx or (
        f"results_two_stage{cu_tag}{re_tag}{dual_tag}_"
        f"{args.dataset}_max{maxmod_label}_seeds{len(seeds)}"
        f"{ti_tag}{pr_tag}{au_tag}{classes_tag}.xlsx"
    )

    from pathlib import Path
    run = setup_run(
        "two_stage_multiclass_greedy_runner",
        args=args, argv=sys.argv,
        name_hint=Path(output_xlsx).stem,
        log_dir=args.log_dir,
        console_level=args.log_level,
        trace_rounds=args.trace_rounds,
        timing=not args.no_fine_timers,
        extra={"resolved_max_modalities": max_modalities,
               "resolved_seeds": list(seeds),
               "resolved_budget_fractions": list(budget_fractions),
               "output_xlsx": output_xlsx},
    )

    t0 = time.time()
    status = "ok"
    all_results = []
    try:
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
            reward_estimate=args.reward_estimate,
            reward_update=args.reward_update,
            alpha_ucb=args.alpha_ucb,
            step_size=args.step_size,
            lambda_max=args.lambda_max,
            run_inference=not args.skip_inference,
            image_pool_side=args.image_pool_side,
            image_data_home=args.image_cache_dir,
            pred_rule=args.pred_rule,
        )
    except BaseException as exc:                          # noqa: BLE001
        # Whatever killed the sweep -- an unguarded bug, Ctrl-C, a PBS
        # walltime signal -- the manifest records it and points at the
        # JSONL, which still holds every row completed up to that moment.
        status = f"failed: {type(exc).__name__}: {exc}"
        log.exception("run aborted after %s", run.rows_path)
        run.finalize(status=status)
        raise

    df = pd.DataFrame(all_results)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    log.info("=" * 70)
    log.info("SUMMARY (mean +/- std across seeds) -- %s", args.dataset)
    log.info("=" * 70)
    log.info("%-10s%-12s%14s%16s%12s%12s%9s", 'Budget', 'InitFrac', 'ErrorRate',
             'TwoStageErr', 'InfErr', 'InfSpent', 'Views')
    log.info("-" * 85)
    for (frac, init_f), sub in ok.groupby(['budget_fraction', 'init_fraction']):
        log.info(
            "%-10.2f%-12.3f%10.3f+/-%.3f%12.3f+/-%.3f%8.3f%12.4f%9.2f",
            frac, init_f,
            sub['error_rate'].mean(), sub['error_rate'].std(),
            sub['two_stage_error'].mean(), sub['two_stage_error'].std(),
            sub['inference_error'].mean(),
            sub['inference_actual_cost'].mean(),
            sub['avg_views_acquired'].mean(),
        )

    log.info("Per-seed wall-clock time (s): %s",
             df.groupby('seed')['seed_time_sec'].first().to_dict())

    save_results_to_excel(all_results, args.dataset, filename=output_xlsx,
                          info_rows=run.info_rows())
    log.info("Execution time: %.1f seconds", time.time() - t0)
    run.finalize(status="ok")