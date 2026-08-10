"""
Driver for gmm_multiclass_submodular.py (the multiclass full-feedback /
bandit-feedback submodular algorithm), structurally mirroring
gmm_submodular_runner.py so multiclass runs slot into
run_proposed_methods.py's unified schema:

  - Data via core.datasets.load_dataset_as_numpy -- same contract as every
    other runner. Works on ANY registered dataset: for the 5 real binary
    datasets and synthetic_asymmetric/synthetic_symmetric, nclasses is
    INFERRED from the labels (= 2, where the multiclass algorithm reduces
    to nearest-mean binary classification); for "synthetic",
    the synthetic_n_classes knob controls how many classes are GENERATED.
  - Costs: fixed per-dataset heterogeneous costs
    (generate_modality_costs_heterogeneous), view 0 free, normalized so
    sum(costs) == 1 -- identical convention to the other runners, NOT the
    notebook's per-trial cost redraw.
  - Split: datasets.split_train_inference per seed (80/20, no val). Each
    seed gets its own split, center init, and inference-phase sampling rng -- this
    REPLACES the notebook's buggy trial loop, where generate_data and
    simulation were re-seeded with the SAME args.seed every trial so all
    trials saw identical data (only costs varied).
  - est_means_init: nclasses distinct rows of X_train, drawn once per seed
    and reused across budget fractions (repo convention, generalizing the
    binary runners' 2-row init; replaces the notebook's rng.normal init).
  - Budget: total_budget = budget_fraction * n_total, train/inference
    split auto-derived as n_train/n_total -- identical to the other
    runners.
  - Inference phase: core.lp_colgen.solve_lp_policy_colgen_multiclass
    (multiclass column generation) -- tractable at any nviews, same as the
    binary colgen path.

`feedback` ("full" | "bandit") selects the training-phase update rule --
see gmm_multiclass_submodular.run_training_phase.

`acquisition` ("greedy" | "lp_chain" | "lp_full") selects the training-phase
ACQUISITION policy, orthogonally to `feedback`:
  greedy    per-round submodular greedy + OMD dual (default; unchanged)
  lp_chain  UCB reward estimates over the nviews+1 nested greedy chain,
            with the per-round budgeted LP solved exactly each round
  lp_full   the same, over the FULL 2^(nviews-1) enumeration -- the
            small-nviews fidelity check for what the chain costs
`reward_update` ("subsets" | "selected") controls whether a round scores
every arm contained in the played subset (counterfactual replay) or only
the played arm. Ignored under acquisition="greedy". See
gmm_multiclass_submodular.py's ACQUISITION MODES docstring section. Results dicts have the
SAME per-fraction key set as gmm_submodular_runner.run_experiment (so
run_proposed_methods._normalize_frac_keyed_results applies unchanged);
feedback / n_classes are experiment-level settings the caller records.
"""

from __future__ import annotations

import time
from pathlib import Path

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
from core.lp_colgen import solve_lp_policy_colgen_multiclass

from gmm_submodular.gmm_multiclass_submodular import (
    ACQUISITION_MODES,
    REWARD_UPDATE_SCOPES,
    greedy_oracle,  # noqa: F401 -- re-exported
    multiclass_risk,  # noqa: F401 -- re-exported
    pred_linear_cla,  # noqa: F401 -- re-exported
    run_training_phase,
    run_inference_phase,
)


def run_experiment(
    dataset_name,
    feedback="full",
    acquisition="greedy",
    reward_update="subsets",
    max_modalities=None,
    seeds=(42, 43, 44, 45, 46),
    budget_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    data_path=None,
    max_samples=None,
    synthetic_n_samples=SYNTHETIC_N_SAMPLES,
    synthetic_n_views=SYNTHETIC_N_VIEWS,
    synthetic_seed=SYNTHETIC_SEED,
    synthetic_mean_scale=SYNTHETIC_MEAN_SCALE,
    synthetic_cluster_std=SYNTHETIC_CLUSTER_STD,
    synthetic_n_classes=SYNTHETIC_N_CLASSES,
    alpha_ucb=2.0,
    lr=1e-2,
    run_inference=True,
    image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
    image_data_home=None,
):
    """
    Multiclass counterpart of gmm_submodular_runner.run_experiment. Same
    return shape: {budget_fraction: {metric_name: [per-seed values]}}.

    feedback: "full" (y_true revealed every round) or "bandit" (one-bit
        reward only) -- selects the training-phase update rule.
    acquisition: "greedy" (default; per-round submodular greedy + OMD dual),
        "lp_chain" (per-round LP over the nviews+1 greedy chain) or
        "lp_full" (per-round LP over the full 2^(nviews-1) enumeration --
        capped by gmm_multiclass_submodular.MAX_REWARD_ESTIMATE_VIEWS).
    reward_update: "subsets" (counterfactual replay of every arm contained
        in the played subset) or "selected" (played arm only). Ignored
        under acquisition="greedy".
    synthetic_n_classes: "synthetic" only. For every other
        dataset, nclasses is inferred from the labels.
    alpha_ucb / lr: training-phase exploration bonus scale and (bandit-only)
        complementary-update learning rate -- notebook defaults.
    run_inference: if False, ONLY the training phase (adaptive online
        training) executes; the inference-phase column-generation LP +
        physical-sampling inference is skipped entirely. The train/inference
        split and budget pools are unchanged (the inference budget is just
        never spent); all inference_* metrics are set to NaN, inference
        spent/time to 0.0, num_masks_inference to NaN, and total_reward
        becomes the TRAIN-ONLY figure (== ph1 train_reward, already the
        accuracy over all n_train rounds) rather than the sample-weighted
        train+inference figure -- so
        a run_inference=False total_reward is NOT directly comparable to a
        full run's. Use it when you only care about training dynamics.
    image_data_home: mnist/fashion_mnist only -- where fetch_openml caches
        its download. See core.datasets.DEFAULT_OPENML_DATA_HOME's comment:
        defaults to a relative "data/openml_cache" folder rather than
        fetch_openml's own ~/scikit_learn_data, since $HOME's quota is tiny
        (and often already exhausted) on clusters like NCI Gadi.
    """
    if dataset_name not in ALL_DATASETS:
        msg = f"dataset_name must be one of {ALL_DATASETS}, got {dataset_name!r}"
        raise ValueError(msg)
    if feedback not in ("full", "bandit"):
        raise ValueError(f"feedback must be 'full' or 'bandit', got {feedback!r}")
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                          f"got {acquisition!r}")
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                          f"got {reward_update!r}")
    if (feedback == "bandit" and reward_update == "subsets"
            and acquisition != "greedy"):
        # Same guard run_training_phase enforces, hoisted here so the run
        # fails before loading a dataset rather than partway through seed 1.
        raise ValueError(
            "feedback='bandit' with reward_update='subsets' is incoherent: the "
            "counterfactual replay reads y_true, which bandit feedback does not "
            "reveal. Use reward_update='selected' with feedback='bandit', or "
            "feedback='full' with reward_update='subsets'.")

    X_full, Y_full, feature_names = load_dataset_as_numpy(
        dataset_name, max_modalities=max_modalities, data_path=data_path,
        max_samples=max_samples, synthetic_n_samples=synthetic_n_samples,
        synthetic_n_views=synthetic_n_views, synthetic_seed=synthetic_seed,
        synthetic_mean_scale=synthetic_mean_scale,
        synthetic_cluster_std=synthetic_cluster_std,
        synthetic_n_classes=synthetic_n_classes,
        image_pool_side=image_pool_side,
        image_data_home=image_data_home,
    )
    n_samples, nviews = X_full.shape
    nclasses = int(Y_full.max()) + 1
    ru_tag = "" if acquisition == "greedy" else f"/{reward_update}"
    print(f"{dataset_name}: {n_samples} samples, {nviews} views, "
          f"{nclasses} classes, feedback={feedback}, "
          f"acquisition={acquisition}{ru_tag} "
          f"(free: '{feature_names[0]}', {nviews - 1} paid)")

    # Fixed per-dataset costs
    raw_feature_costs = np.array(
        generate_modality_costs_heterogeneous(n_features=nviews, dataset_name=dataset_name),
        dtype=np.float64,
    )
    costs = raw_feature_costs / raw_feature_costs.sum()
    paid_costs = costs[1:]
    print(f"feature_costs[0] (free) = {costs[0]}, paid costs (normalized, sum(costs)==1): "
          f"min={paid_costs.min():.4f}, max={paid_costs.max():.4f}, "
          f"mean={paid_costs.mean():.4f}, sum={paid_costs.sum():.4f}")

    results = {
        frac: {"train_reward": [], "train_f1": [], "train_auroc": [],
               "inference_reward": [], "inference_f1": [], "inference_auroc": [],
               "total_reward": [],
               "train_spent": [], "inference_spent": [], "num_masks_inference": [],
               "train_time_sec": [], "inference_time_sec": [], "seed_time_sec": [],
               "n_train": [], "n_inference": [],
               "train_budget": [], "inference_budget": [],
               "n_arms": [], "avg_views_train": []}
        for frac in budget_fractions
    }

    for seed in seeds:
        seed_start = time.time()
        print(f"\n{'=' * 60}\n=== SEED {seed} ({dataset_name}, {feedback} feedback, "
              f"{acquisition} acquisition) ===\n{'=' * 60}")

        train_idx, test_idx = split_train_inference(n_samples, seed=seed)
        X_train, Y_train = X_full[train_idx], Y_full[train_idx]
        X_inf, Y_inf = X_full[test_idx], Y_full[test_idx]
        n_train, n_inf = len(train_idx), len(test_idx)
        n_total = n_train + n_inf

        rng = np.random.default_rng(seed=seed)  # warmup preds, inference-phase sampling, center init

        # nclasses distinct rows of X_train as initial centers -- repo
        # convention (generalizes the binary runners' 2-row init). Drawn
        # once per seed, reused across budget fractions.
        #n_seed_rows = min(nclasses, len(X_train))
        #seed_rows = rng.choice(len(X_train), size=n_seed_rows, replace=False)
        #est_means_init = X_train[seed_rows[np.arange(nclasses) % n_seed_rows]].astype(float).copy()
        est_means_init = rng.normal(size=(nclasses, nviews))

        for frac in budget_fractions:
            total_budget = frac * n_total
            train_inference_split = n_train / n_total
            training_budget = train_inference_split * total_budget
            inference_budget = total_budget - training_budget
            print(f"\nBudget fraction {frac:.2f} -> total_budget={total_budget:.4f} (n_total={n_total})")
            print(f"  Train pool: {training_budget:.4f} | Inference pool: {inference_budget:.4f}")

            train_start = time.time()
            ph1 = run_training_phase(
                nviews=nviews, nclasses=nclasses, costs=costs, n_train=n_train,
                training_budget=training_budget, X_train=X_train, Y_train=Y_train,
                est_means_init=est_means_init, feedback=feedback,
                alpha_ucb=alpha_ucb, lr=lr, rng=rng,
                acquisition=acquisition, reward_update=reward_update,
            )
            train_time = time.time() - train_start
            print(f"  [Training] Reward: {ph1['train_reward']:.3f} | F1: {ph1['train_f1']:.3f} "
                  f"| AUROC: {ph1['train_auroc']:.3f} | Spent: {ph1['spent']:.4f} / {training_budget:.4f} "
                  f"| Time: {train_time:.2f}s")
            if acquisition != "greedy":
                print(f"  [Training] Acquisition {acquisition}/{reward_update}: "
                      f"{ph1['n_arms']} arms "
                      f"(full enumeration would have been 2^{nviews - 1} = "
                      f"{2 ** (nviews - 1)}) | avg views/round: "
                      f"{ph1['avg_views_acquired']:.2f} | distinct masks played: "
                      f"{ph1['n_unique_masks']}")

            if run_inference:
                inference_start = time.time()
                masks, probs = solve_lp_policy_colgen_multiclass(
                    est_means=ph1["est_means"], costs=costs,
                    inference_budget=inference_budget, n_inference=n_inf,
                )
                combo_costs = np.array([np.sum(costs[m]) for m in masks])
                print(f"  [Inference] Column generation discovered {len(masks)} subsets "
                      f"(power set would have been 2^{nviews}-1 = {2**nviews - 1})")
                ph2 = run_inference_phase(
                    masks=masks, probs=probs, combo_costs=combo_costs, costs=costs,
                    est_means=ph1["est_means"],
                    inference_budget=inference_budget, X_inf=X_inf, Y_inf=Y_inf, rng=rng,
                )
                inference_time = time.time() - inference_start
                print(f"  [Inference] Reward: {ph2['inference_reward']:.3f} | F1: {ph2['inference_f1']:.3f} "
                      f"| AUROC: {ph2['inference_auroc']:.3f} | Spent: {ph2['spent']:.4f} / {inference_budget:.4f} "
                      f"| Time: {inference_time:.2f}s")

                total_reward = (n_train * ph1["train_reward"] + n_inf * ph2["inference_reward"]) / n_total
                num_masks = len(masks)
            else:
                # --skip-inference: inference-phase column-gen + sampling not
                # run. inference_* -> NaN, spent/time -> 0, num_masks -> NaN,
                # and total_reward is the TRAIN-ONLY figure (ph1 train_reward,
                # already the accuracy over all n_train rounds). See
                # run_experiment's docstring.
                ph2 = {
                    "inference_reward": float("nan"),
                    "inference_f1": float("nan"),
                    "inference_auroc": float("nan"),
                    "spent": 0.0,
                }
                inference_time = 0.0
                num_masks = np.nan
                total_reward = ph1["train_reward"]
                print(f"  [Inference] SKIPPED (--skip-inference); train-only total_reward")

            print(f"  [Total]   Reward: {total_reward:.3f}")

            results[frac]["train_reward"].append(ph1["train_reward"])
            results[frac]["train_f1"].append(ph1["train_f1"])
            results[frac]["train_auroc"].append(ph1["train_auroc"])
            results[frac]["inference_reward"].append(ph2["inference_reward"])
            results[frac]["inference_f1"].append(ph2["inference_f1"])
            results[frac]["inference_auroc"].append(ph2["inference_auroc"])
            results[frac]["total_reward"].append(total_reward)
            results[frac]["n_arms"].append(ph1["n_arms"])
            results[frac]["avg_views_train"].append(ph1["avg_views_acquired"])
            results[frac]["train_spent"].append(ph1["spent"])
            results[frac]["inference_spent"].append(ph2["spent"])
            results[frac]["num_masks_inference"].append(num_masks)
            results[frac]["train_time_sec"].append(train_time)
            results[frac]["inference_time_sec"].append(inference_time)
            results[frac]["n_train"].append(n_train)
            results[frac]["n_inference"].append(n_inf)
            results[frac]["train_budget"].append(training_budget)
            results[frac]["inference_budget"].append(inference_budget)

        seed_elapsed = time.time() - seed_start
        print(f"\n  [SEED {seed}] wall-clock time: {seed_elapsed:.1f}s "
              f"({len(budget_fractions)} budget fractions)")
        for frac in budget_fractions:
            results[frac]["seed_time_sec"].append(seed_elapsed)

    return results


def save_results_to_excel(results, budget_fractions, dataset_name, feedback,
                          seeds=None, filename=None, acquisition="greedy",
                          reward_update=""):
    """Same shape as gmm_submodular_runner.save_results_to_excel, plus
    'Feedback' / 'Acquisition' / 'Reward Update' columns on every row
    (experiment-level settings, constant within one file, recorded per row
    so files concatenate cleanly)."""
    if filename is None:
        filename = f"results_gmm_{feedback}_{acquisition}_{dataset_name}.xlsx"

    detailed_rows = []
    for frac in budget_fractions:
        r = results[frac]
        n = len(r["train_reward"])
        row_labels = seeds if seeds is not None and len(seeds) == n else range(n)
        for label, i in zip(row_labels, range(n)):
            detailed_rows.append({
                "Seed": label,
                "Feedback": feedback,
                "Acquisition": acquisition,
                "Reward Update": reward_update,
                "Budget Fraction": frac,
                "Train Reward": r["train_reward"][i],
                "Train F1": r["train_f1"][i],
                "Train AUROC": r["train_auroc"][i],
                "Inference Reward": r["inference_reward"][i],
                "Inference F1": r["inference_f1"][i],
                "Inference AUROC": r["inference_auroc"][i],
                "Total Reward": r["total_reward"][i],
                "Total Error": 1 - r["total_reward"][i],
                "Train Spent": r["train_spent"][i],
                "Inference Spent": r["inference_spent"][i],
                "Num Masks Inference": r["num_masks_inference"][i],
                "Train Time (s)": r["train_time_sec"][i],
                "Inference Time (s)": r["inference_time_sec"][i],
                "Seed Time (s)": r["seed_time_sec"][i],
                "Train Samples": r["n_train"][i],
                "Inference Samples": r["n_inference"][i],
                "Train Budget": r["train_budget"][i],
                "Inference Budget": r["inference_budget"][i],
                "Num Arms": r["n_arms"][i] if "n_arms" in r else np.nan,
                "Avg Views Train": (r["avg_views_train"][i]
                                    if "avg_views_train" in r else np.nan),
            })
    df_detailed = pd.DataFrame(detailed_rows)

    summary_rows = []
    for frac in budget_fractions:
        r = results[frac]
        summary_rows.append({
            "Budget Fraction": frac,
            "Feedback": feedback,
            "Acquisition": acquisition,
            "Reward Update": reward_update,
            "Train Reward Mean": np.mean(r["train_reward"]),
            "Train Reward Std": np.std(r["train_reward"]),
            "Train F1 Mean": np.nanmean(r["train_f1"]),
            "Train F1 Std": np.nanstd(r["train_f1"]),
            "Train AUROC Mean": np.nanmean(r["train_auroc"]),
            "Train AUROC Std": np.nanstd(r["train_auroc"]),
            "Inference Reward Mean": np.mean(r["inference_reward"]),
            "Inference Reward Std": np.std(r["inference_reward"]),
            "Inference F1 Mean": np.nanmean(r["inference_f1"]),
            "Inference F1 Std": np.nanstd(r["inference_f1"]),
            "Inference AUROC Mean": np.nanmean(r["inference_auroc"]),
            "Inference AUROC Std": np.nanstd(r["inference_auroc"]),
            "Total Reward Mean": np.mean(r["total_reward"]),
            "Total Reward Std": np.std(r["total_reward"]),
            "Total Error Mean": 1 - np.mean(r["total_reward"]),
            "Train Spent Mean": np.mean(r["train_spent"]),
            "Inference Spent Mean": np.mean(r["inference_spent"]),
            "Num Masks Inference Mean": np.mean(r["num_masks_inference"]),
            "Train Time Mean (s)": np.mean(r["train_time_sec"]),
            "Inference Time Mean (s)": np.mean(r["inference_time_sec"]),
            "Seed Time Mean (s)": np.mean(r["seed_time_sec"]),
            "Train Samples": r["n_train"][0],
            "Inference Samples": r["n_inference"][0],
            "Train Budget Mean": np.mean(r["train_budget"]),
            "Inference Budget Mean": np.mean(r["inference_budget"]),
            "Num Arms": (r["n_arms"][0] if r.get("n_arms") else np.nan),
            "Avg Views Train Mean": (np.mean(r["avg_views_train"])
                                     if r.get("avg_views_train") else np.nan),
        })
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_detailed.to_excel(writer, sheet_name="Detailed Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(filename)
    _style_sheet(wb["Detailed Results"])
    _style_sheet(wb["Summary"])
    wb.save(filename)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the multiclass submodular AFA algorithm (full or bandit feedback) "
                     "on a dataset, with the same training/inference-phase and budget conventions "
                     "as the other runners."
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--feedback", choices=("full", "bandit"), default="full")
    parser.add_argument("--acquisition", choices=ACQUISITION_MODES, default="greedy",
                         help="Training-phase acquisition policy. 'greedy' (default) is the "
                              "per-round submodular greedy + OMD dual, unchanged. 'lp_chain' "
                              "keeps UCB reward estimates over the nviews+1 nested greedy "
                              "chain and solves the per-round budgeted LP over them exactly. "
                              "'lp_full' does the same over the FULL 2^(nviews-1) enumeration "
                              "-- the small-nviews fidelity check for what the chain costs.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                         help="LP acquisition modes only: 'subsets' (default) replays every "
                              "arm contained in the played subset (counterfactual, needs the "
                              "true label); 'selected' credits only the played arm from the "
                              "1-bit reward. Ignored under --acquisition greedy.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-modalities", type=str, default="all",
                         help="Integer, or 'all' (default). greedy_oracle scales fine to full "
                              "feature counts, same as the binary submodular runner.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES)
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS)
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                         help="synthetic only: how many classes to generate. "
                              "Every other dataset infers nclasses from its labels.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED)
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE,
                         help="Per-(class,view) means ~ Uniform(0, mean_scale). The multiclass "
                              "notebook's snrdb=9 corresponds to mean_scale=10**(9/20)~=2.818.")
    parser.add_argument("--alpha-ucb", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-2,
                         help="bandit feedback only: complementary-label update learning rate.")
    parser.add_argument("--skip-inference", action="store_true",
                         help="Run ONLY the training phase (adaptive online training); skip the "
                              "inference-phase column-generation LP inference. Split and budgets are unchanged; "
                              "all inference_* columns become NaN and total_reward becomes the "
                              "train-only figure. Output filename gets a _trainonly tag.")
    parser.add_argument("--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE,
                         help=f"mnist/fashion_mnist only: block-average each 28x28 image down "
                              f"to x*x features (x=this value; default {DEFAULT_IMAGE_POOL_SIDE} "
                              f"-> {DEFAULT_IMAGE_POOL_SIDE**2} features). Pass 28 to keep all "
                              f"784 raw pixels. Ignored for non-image datasets.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                         help="mnist/fashion_mnist only: directory fetch_openml caches its "
                              "download in. Default (None) resolves to core.datasets."
                              "DEFAULT_OPENML_DATA_HOME, a RELATIVE 'data/openml_cache' folder -- "
                              "deliberately NOT fetch_openml's own ~/scikit_learn_data default, "
                              "which exceeds $HOME's quota on clusters like NCI Gadi. Point this "
                              "at your project/scratch space if the default location itself lacks "
                              "quota. Ignored for non-image datasets.")
    parser.add_argument("--output-xlsx", type=str, default=None)
    args = parser.parse_args()

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    t0 = time.time()
    results = run_experiment(
        args.dataset,
        feedback=args.feedback,
        acquisition=args.acquisition,
        reward_update=args.reward_update,
        max_modalities=max_modalities,
        seeds=seeds,
        budget_fractions=budget_fractions,
        data_path=args.data_path,
        max_samples=args.max_samples,
        synthetic_n_samples=args.n_samples,
        synthetic_n_views=args.n_views,
        synthetic_seed=args.synthetic_seed,
        synthetic_mean_scale=args.mean_scale,
        synthetic_n_classes=args.num_classes,
        alpha_ucb=args.alpha_ucb,
        lr=args.lr,
        run_inference=not args.skip_inference,
        image_pool_side=args.image_pool_side,
        image_data_home=args.image_cache_dir,
    )

    acq_label = args.acquisition + ("" if args.acquisition == "greedy"
                                    else f"/{args.reward_update}")
    print(f"\n{'=' * 70}\nSUMMARY (mean +/- std across seeds) -- {args.dataset} / "
          f"{args.feedback} / {acq_label}\n{'=' * 70}")
    print(f"{'Fraction':<10}{'Train Reward':>16}{'Train F1':>12}{'Train AUROC':>14}"
          f"{'Inf Reward':>14}{'Inf F1':>10}{'Inf AUROC':>12}{'Total Reward':>16}{'Total Error':>14}")
    print("-" * 130)
    for frac in budget_fractions:
        r = results[frac]
        print(
            f"{frac:<10.1f}"
            f"{np.mean(r['train_reward']):>11.3f}+/-{np.std(r['train_reward']):.3f}"
            f"{np.nanmean(r['train_f1']):>7.3f}"
            f"{np.nanmean(r['train_auroc']):>14.3f}"
            f"{np.mean(r['inference_reward']):>9.3f}"
            f"{np.nanmean(r['inference_f1']):>10.3f}"
            f"{np.nanmean(r['inference_auroc']):>12.3f}"
            f"{np.mean(r['total_reward']):>11.3f}+/-{np.std(r['total_reward']):.3f}"
            f"{1 - np.mean(r['total_reward']):>10.3f}"
        )

    ti_tag = "_trainonly" if args.skip_inference else ""
    # max_modalities is None when --max-modalities all was passed (no
    # truncation) -- render that as "ALL" instead of the literal "maxNone".
    maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
    # Synthetic datasets only: include the actual class count in the
    # filename (--num-classes for synthetic; fixed at 2 for the
    # binary synthetic generators) so runs at different K don't overwrite
    # each other. Real datasets aren't tagged -- their class count is fixed
    # by the data itself, not a run parameter.
    classes_tag = ""
    if args.dataset in SYNTHETIC_DATASETS:
        n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
        classes_tag = f"_classes{n_classes}"
    # Acquisition tag so greedy / lp_chain / lp_full runs on the same dataset
    # do not overwrite each other. Empty for the default, keeping existing
    # filenames byte-identical to before this option existed.
    acq_tag = ("" if args.acquisition == "greedy"
               else f"_{args.acquisition}_{args.reward_update}")
    output_xlsx = args.output_xlsx or (
        f"results_gmm_{args.feedback}{acq_tag}_{args.dataset}_"
        f"max{maxmod_label}_seeds{len(seeds)}{ti_tag}{classes_tag}.xlsx"
    )
    save_results_to_excel(results, budget_fractions, args.dataset, args.feedback,
                          seeds=seeds, filename=output_xlsx,
                          acquisition=args.acquisition,
                          reward_update=("" if args.acquisition == "greedy"
                                         else args.reward_update))
    print(f"Execution time: {time.time() - t0:.1f} seconds")