"""
run_proposed_methods.py

Single CLI entry point that runs any ONE of the proposed MULTICLASS AFA
methods -- each of which works on any registered dataset (the real
AFA-Benchmark datasets, where nclasses is inferred from the labels, or the
synthetic GMM datasets including synthetic) --

    --method submodular       -> gmm_multiclass_submodular_runner.run_experiment
    --method two_stage_greedy -> two_stage_multiclass_greedy_runner.run_experiment

-- and writes ONE Excel file with a unified row schema across both, so
results from different methods can be concatenated/compared directly
instead of living in differently-shaped workbooks.

=== The EXP4 `two_stage` method has been REMOVED ===
two_stage/two_stage_multiclass.py (the EXP4/Hedge-BwK Stage-2 bandit over
view-combination experts) and its runner were deleted from the tree, so
--method two_stage, --warm-start and --gamma-max are gone with them; the
only Stage-1-based method left is two_stage_greedy. The parts of that
module that were never EXP4-specific -- Stage-1 centre init, the two
prediction rules, the macro metrics, the LP inference routine -- moved
VERBATIM to core/multiclass_common.py and are still used by
two_stage_greedy. `Warm Start` stays in the unified column set (always
False now) so old and new workbooks still concatenate.

=== Multiclass-only ===
The original binary methods (--method bandit -> gmm_2class_bandit,
--method submodular -> gmm_2class_submodular, --method two_stage ->
two_stage_asymmetric) have been REMOVED from this entry point: their
algorithms are hardcoded to K=2 (2-row centers, binary prediction rules,
scalar "evidence for class 1" AUROC) and would silently mis-handle labels
outside {0, 1}. The two methods kept here are both class-count-agnostic:
they infer nclasses from the data (or generate it via --num-classes for
synthetic) and reduce cleanly to the binary case at K == 2, so
they still run correctly on the 2-class datasets too.

  - submodular: gmm_submodular/gmm_multiclass_submodular.py, runnable in
    full-feedback or bandit-feedback mode via --feedback, and under any of
    four ACQUISITION policies via --acquisition (greedy / lp_chain /
    lp_full / lp_full_opt -- see that module's ACQUISITION MODES docstring
    section).
    --acquisition is orthogonal to --feedback: the former picks which
    subset is acquired each round, the latter how the centres are updated
    afterwards.
  - two_stage_greedy: two_stage_greedy/two_stage_multiclass_greedy.py --
    Stage-1 supervised centre init, then a Stage-2 submodular-greedy (or
    UCB-over-arms) acquisition policy in place of EXP4's experts.

This file does NOT change any algorithm behavior. It only calls the two
run_experiment functions (unchanged) and reshapes their already-computed
metrics into one common table. Both compute the same core metric set:
train/inference reward (accuracy), train/inference F1 (macro), train/
inference AUROC (macro one-vs-rest), train/inference spent cost, train/
inference wall time, and per-seed wall time.

=== Unified columns ===
Method, Feedback (NaN for two_stage -- the full/bandit distinction is a
submodular concept), Acquisition (each method's defining policy knob, as
one string: the --acquisition mode (+ --reward-update scope) for
both methods, now spelled IDENTICALLY in each -- previously
two_stage_greedy's defining knob lived only in the auto-named filename, so
concatenated workbooks could not tell two variants apart), Num Classes, Warm Start (False for submodular --
no warm-start concept), Dataset, Seed, Budget Fraction, Init Fraction (NaN
for submodular -- it has no init-fraction sweep), Train Reward, Train
F1, Train AUROC, Inference Reward, Inference F1, Inference AUROC, Total
Reward, Two Stage Error / Init Error (two_stage_greedy only), Train Samples,
Inference Samples, Train Budget, Inference Budget, Train Spent, Inference
Spent, Train Time (s), Inference Time (s), Seed Time (s), Num Masks Inference.

=== How "Total Reward" is computed for two_stage ===
submodular computes a single sample-weighted train+inference reward
(n_train*train_reward + n_inf*inference_reward)/n_total in 0/1-accuracy
units. two_stage doesn't have a single "train_reward" over all of
n_train -- its training split is itself divided into a T1 init stage
(Stage 1, accuracy = 1 - init_error) and a T2 bandit-training stage
(Stage 2, accuracy = avg_reward), with T1 + T2 == n_train.
two_stage_multiclass_runner.py's Total Reward combines all three pieces
with the SAME sample-weighting convention: ((1-init_error)*T1 + avg_reward*
T2 + inference_accuracy*n_inference) / (n_train + n_inference), directly
comparable to submodular's Total Reward. two_stage_error (blending ONLY the
init and bandit-training errors, NOT inference) is kept in its own
"Two Stage Error" column, not folded into Total Reward.

=== Usage ===
    python run_proposed_methods.py --method submodular --dataset synthetic --num-classes 4
    python run_proposed_methods.py --method submodular --dataset ckd --feedback bandit
    python run_proposed_methods.py --method submodular --dataset synthetic --acquisition lp_chain
    python run_proposed_methods.py --method submodular --dataset synthetic --acquisition lp_full --reward-update selected --max-modalities 8
    python run_proposed_methods.py --method two_stage_greedy --dataset synthetic --num-classes 3 --max-modalities 10
    python run_proposed_methods.py --method two_stage_greedy --dataset actg175 --acquisition lp_chain
    # oracle ceiling for lp_full (synthetic only -- it needs the generative means):
    python run_proposed_methods.py --method submodular --dataset synthetic --acquisition lp_full_opt --max-modalities 8

Run `python run_proposed_methods.py --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from core.datasets import (
    ALL_DATASETS,
    DATASET_N_CLASSES,
    DEFAULT_IMAGE_POOL_SIDE,
    MULTICLASS_SYNTHETIC_DATASETS,
    SYNTHETIC_DATASETS,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_SAMPLES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
)
from core.excel_utils import _style_sheet, serialize_selected_subsets

import gmm_submodular.gmm_multiclass_submodular_runner as multiclass_runner
import two_stage_greedy.two_stage_multiclass_greedy_runner as two_stage_mc_greedy_runner
from core.multiclass_common import PRED_RULES

# --acquisition and --reward-update are SHARED by both methods and mean the
# same thing in each, so both read from ONE definition in core rather than
# from per-method copies the driver would have to assert were in sync.
from core.submodular_greedy import (
    ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
    REWARD_UPDATE_SCOPES,
)

# Both methods are MULTICLASS and class-count-agnostic (they reduce to the
# binary case at K == 2). The binary-only bandit / submodular / two_stage
# methods have been removed from this entry point -- see module docstring.
METHODS = ("submodular", "two_stage_greedy")

# The two_stage FAMILY: Stage-1 centre init + init_fraction sweep + the
# shared row schema. Now a family of ONE, since the EXP4 `two_stage` method
# was deleted; kept as a tuple (rather than inlined as a == comparison)
# because the flag guards below read more clearly as membership tests and
# because a second Stage-1-based method would rejoin it here.
TWO_STAGE_FAMILY = ("two_stage_greedy",)

# Both methods now expose the SAME acquisition axis (--acquisition) and the
# same arm-scoring scope (--reward-update), running the same
# core.submodular_greedy helpers underneath. They differ only in the centre
# axis: submodular's centres learn (--feedback), two_stage_greedy's are
# frozen after Stage 1. So there is no longer a "which methods share this
# flag" partition to maintain -- the old UCB_GREEDY_FAMILY constant is gone.

# Each method's own default for --max-modalities when the user doesn't
# override it (matches that method's own __main__ default -- two_stage
# needs truncation for tractability because its Stage-2 EXP4 bandit
# enumerates 2^(nviews-1) experts; submodular's greedy oracle scales to
# full feature counts, so it defaults to no truncation).
# two_stage_greedy enumerates NOTHING (its greedy oracle is O(nviews^2)
# per round), so like submodular it needs no truncation for training --
# though its Stage-2 LP pricing is still the practical ceiling (~16-18 views).
DEFAULT_MAX_MODALITIES = {"submodular": None, "two_stage_greedy": None}


# ─────────────────────────────────────────────────────────────────────────
# Normalization: reshape each method's native results into one common
# row schema (a list of flat dicts), ready for pd.DataFrame(rows).
# ─────────────────────────────────────────────────────────────────────────
def _normalize_frac_keyed_results(results, budget_fractions, seeds, dataset_name, method_name,
                                   feedback=np.nan, n_classes=2,
                                   acquisition=np.nan, alpha_ucb=np.nan):
    """Normalizer for submodular's results dict, which has the shape
    {budget_fraction: {metric_name: [per-seed values]}}.

    feedback / n_classes / acquisition / alpha_ucb: experiment-level
    settings recorded per row. alpha_ucb is taken from the caller because
    gmm_multiclass_submodular_runner does not put it in its results dict
    (unlike the two_stage runner, whose rows carry their own); it is the
    same value for every row of one run either way."""
    rows = []
    for frac in budget_fractions:
        r = results[frac]
        n = len(r["train_reward"])
        row_labels = seeds if len(seeds) == n else range(n)
        for label, i in zip(row_labels, range(n)):
            row = {
                "Method": method_name,
                "Feedback": feedback,
                "Acquisition": acquisition,
                "Alpha UCB": alpha_ucb,
                "Num Classes": n_classes,
                "Warm Start": False,  # submodular has no warm-start concept
                "Dataset": dataset_name,
                "Seed": label,
                "Budget Fraction": frac,
                "Init Fraction": np.nan,
                "Train Reward": r["train_reward"][i],
                "Train F1": r["train_f1"][i],
                "Train AUROC": r["train_auroc"][i],
                "Inference Reward": r["inference_reward"][i],
                "Inference F1": r["inference_f1"][i],
                "Inference AUROC": r["inference_auroc"][i],
                "Total Reward": r["total_reward"][i],
                "Two Stage Error": np.nan,
                "Init Error": np.nan,
                "Train Spent": r["train_spent"][i],
                "Inference Spent": r["inference_spent"][i],
                "Train Time (s)": r["train_time_sec"][i],
                "Inference Time (s)": r["inference_time_sec"][i],
                "Seed Time (s)": r["seed_time_sec"][i],
                "Num Masks Inference": r["num_masks_inference"][i] if "num_masks_inference" in r else np.nan,
                "Train Samples": r["n_train"][i],
                "Inference Samples": r["n_inference"][i],
                "Train Budget": r["train_budget"][i],
                "Inference Budget": r["inference_budget"][i],
                "Num Arms": r["n_arms"][i] if "n_arms" in r else np.nan,
                "Selected Subsets": serialize_selected_subsets(
                    r["selected_subsets"][i]),
            }
            rows.append(row)
    return rows


def normalize_two_stage(all_results, dataset_name,
                        method_name="two_stage_greedy", acquisition=np.nan,
                        alpha_ucb=np.nan):
    """Normalizer for two_stage_multiclass_runner's flat list-of-dicts
    results. Num Classes is read from each row's own 'nclasses' (inferred
    from the labels, or --num-classes for synthetic), NOT
    hardcoded.

    alpha_ucb: fallback for the 'Alpha UCB' column; each row's own
    'alpha_ucb' is preferred when the runner recorded one, so a caller
    that sweeps alpha per row is reported correctly.

    acquisition: the run's acquisition policy as one string, for the
    shared 'Acquisition' column -- two_stage_greedy passes
the same "greedy" / "lp_chain+subsets" / "lp_full+selected" strings
    submodular reports, since both methods now share the axis."""
    rows = []
    for d in all_results:
        rows.append({
            "Method": method_name,
            "Feedback": np.nan,
            "Acquisition": acquisition,
            "Alpha UCB": d.get("alpha_ucb", alpha_ucb),
            "Num Classes": d.get("nclasses", np.nan),
            "Warm Start": bool(d.get("warm_start", False)),
            "Dataset": dataset_name,
            "Seed": d["seed"],
            "Budget Fraction": d["budget_fraction"],
            "Init Fraction": d["init_fraction"],
            "Train Reward": d["avg_reward"],
            "Train F1": d["train_f1"],
            "Train AUROC": d["train_auroc"],
            "Inference Reward": d["inference_accuracy"],
            "Inference F1": d["inference_f1"],
            "Inference AUROC": d["inference_auroc"],
            "Total Reward": d["total_reward"],
            "Two Stage Error": d["two_stage_error"],
            "Init Error": d["init_error"],
            "Train Spent": d["training_budget_spent"],
            "Inference Spent": d["inference_actual_cost"],
            "Train Time (s)": d["train_time_sec"],
            "Inference Time (s)": d["inference_time_sec"],
            "Seed Time (s)": d["seed_time_sec"],
            "Num Masks Inference": d.get("num_masks_inference", np.nan),
            "Train Samples": d["n_train"],
            "Inference Samples": d["n_test"],
            "Train Budget": d["training_budget"],
            "Inference Budget": d["inference_budget"],
            "Num Arms": d.get("n_arms", np.nan),
            "Selected Subsets": d.get(
                "Selected Subsets",
                serialize_selected_subsets(d.get("selected_subsets", [])),
            ),
        })
    return rows


UNIFIED_COLUMNS = [
    "Method", "Feedback", "Num Classes", "Warm Start", "Dataset", "Seed",
    "Budget Fraction", "Init Fraction",
    "Train Reward", "Train F1", "Train AUROC",
    "Inference Reward", "Inference F1", "Inference AUROC",
    "Total Reward", "Two Stage Error", "Init Error",
    "Train Samples", "Inference Samples",
    "Train Budget", "Inference Budget",
    "Train Spent", "Inference Spent",
    "Train Time (s)", "Inference Time (s)", "Seed Time (s)",
    "Num Masks Inference",
    # Appended at the END, deliberately: prepending would reorder every
    # existing workbook's columns and break side-by-side diffs of old and
    # new result files.
    "Acquisition", "Num Arms",
    # Recorded from this run's --alpha-ucb. Previously the value survived
    # ONLY in the auto-generated filename, so two runs differing just in
    # alpha were indistinguishable once inside the workbook.
    "Alpha UCB",
    "Selected Subsets",
]


def _acquisition_label(acquisition, reward_update):
    """One string naming the run's acquisition policy, for the shared
    'Acquisition' column and the auto-named filename tag. Now identical
    across BOTH methods, which is the point of the unification:
    "greedy", "lp_chain+subsets", "lp_full+selected", ...
    """
    if acquisition in ("greedy",) + ORACLE_ACQUISITION_MODES:
        # reward_update is inert under BOTH: greedy has no enumerated arms
        # to score, and the oracle modes have exact arm values that are
        # never scored. Appending a scope either would advertise a scoring
        # rule that never ran -- and would split one policy across two
        # 'Acquisition' values in the unified table purely on the strength
        # of a flag that did nothing.
        return acquisition
    return f"{acquisition}+{reward_update}"


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────
def run_method(method, dataset, max_modalities, seeds, budget_fractions,
                data_path, n_init_fraction_points, max_samples,
                synthetic_n_samples, synthetic_seed, synthetic_mean_scale,
                feedback="full",
                synthetic_n_classes=SYNTHETIC_N_CLASSES,
                step_size=1.0, lambda_max=10.0,
                run_inference=True, image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
                image_data_home=None, pred_rule="nearest_center",
                reward_update="subsets", alpha_ucb=2.0, lr=1e-2,
                acquisition="greedy"):
    # For synthetic datasets, --max-modalities IS the view count to
    # generate (no separate --n-views knob -- they were doing the same
    # conceptual job: "how many features/views to use"). 'all'
    # (max_modalities=None) has no natural meaning for something with no
    # fixed width to truncate, so it falls back to SYNTHETIC_N_VIEWS.
    synthetic_n_views = max_modalities if max_modalities is not None else SYNTHETIC_N_VIEWS

    common_kwargs = dict(
        max_modalities=max_modalities, seeds=seeds, budget_fractions=budget_fractions,
        data_path=data_path, max_samples=max_samples,
        synthetic_n_samples=synthetic_n_samples, synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed, synthetic_mean_scale=synthetic_mean_scale,
    )

    # Flag applicability. --acquisition, --reward-update, --alpha-ucb and
    # --skip-inference apply to BOTH methods, so none of them needs a "wrong
    # method" warning any more. What is left to flag is (a) flags that are
    # inert for the CHOSEN acquisition mode, and (b) the centre-axis flags
    # (--feedback, --lr), which only submodular has -- two_stage_greedy
    # holds its Stage-1 centres frozen under every acquisition mode.
    # step_size / lambda_max only warn when set AWAY from their defaults, so
    # a plain submodular run doesn't emit spurious warnings.
    if acquisition == "greedy" and reward_update != "subsets":
        print("WARNING: --reward-update has no effect under --acquisition greedy "
              "(there are no enumerated arms to score); ignored.")
    if acquisition in ORACLE_ACQUISITION_MODES and reward_update != "subsets":
        print(f"WARNING: --reward-update has no effect under --acquisition "
              f"{acquisition} (its arm values come from the true means and are "
              f"never scored); ignored.")
    if acquisition in ORACLE_ACQUISITION_MODES and method == "submodular" and feedback != "full":
        # NOT an error: the oracle governs ACQUISITION, --feedback governs the
        # CENTRE update, and both still run. Worth saying out loud because the
        # combination is easy to read as "fully oracle" when it is not.
        print(f"NOTE: --acquisition {acquisition} gives the ACQUISITION policy "
              f"the true means; the classifier still learns under --feedback "
              f"{feedback}. This is not an oracle-classifier run.")
    if method != "submodular" and feedback != "full":
        print("WARNING: --feedback governs the CENTRE update, which only "
              "submodular has (two_stage_greedy holds its Stage-1 centres "
              f"frozen); ignored for {method!r}. The arm-scoring analogue is "
              "--reward-update subsets/selected.")
    if method != "submodular" and lr != 1e-2:
        print("WARNING: --lr is the complementary-label step in submodular's "
              f"--feedback bandit centre update; ignored for {method!r}, whose "
              "centres are frozen.")
    if method not in TWO_STAGE_FAMILY:
        if (step_size, lambda_max) != (1.0, 10.0):
            print("WARNING: --step-size / --lambda-max only apply to "
                  f"{TWO_STAGE_FAMILY}; ignored for {method!r}.")
        if pred_rule != "nearest_center":
            print(f"WARNING: --pred-rule only applies to --method two_stage_greedy; "
                  f"ignored for {method!r} (submodular always uses its own "
                  f"pairwise-vote rule, which is what --pred-rule pairwise_vote "
                  f"makes two_stage match).")

    if method == "submodular":
        results = multiclass_runner.run_experiment(
            dataset, feedback=feedback,
            acquisition=acquisition, reward_update=reward_update,
            # alpha_ucb / lr were previously NOT forwarded here even though
            # gmm_multiclass_submodular_runner.run_experiment has always
            # accepted them -- so --alpha-ucb 3 was silently ignored for
            # this method. Forwarding them is behaviour-preserving at the
            # defaults, which are identical on both sides (2.0 / 1e-2).
            alpha_ucb=alpha_ucb, lr=lr,
            synthetic_n_classes=synthetic_n_classes,
            run_inference=run_inference, image_pool_side=image_pool_side,
            image_data_home=image_data_home,
            **common_kwargs
        )
        # Actual class count for the reported "Num Classes" column. Prefer
        # the DATASET_N_CLASSES lookup (covers real multiclass datasets like
        # diabetes=3, mnist/fashion_mnist=10); for the multiclass SYNTHETIC
        # dataset it's whatever synthetic_n_classes was generated; else 2.
        if dataset in MULTICLASS_SYNTHETIC_DATASETS:
            n_classes = synthetic_n_classes
        else:
            n_classes = DATASET_N_CLASSES.get(dataset, 2)
        return _normalize_frac_keyed_results(
            results, budget_fractions, seeds, dataset, "submodular",
            feedback=feedback, n_classes=n_classes,
            acquisition=_acquisition_label(acquisition, reward_update),
            alpha_ucb=alpha_ucb,
        )

    if method == "two_stage_greedy":
        # --acquisition is passed STRAIGHT THROUGH: the runner and
        # run_alg_greedy_multiclass take the same three mode names this
        # flag offers, so there is nothing to translate. `lr` is not
        # forwarded -- it is submodular's complementary-label step, and
        # two_stage_greedy's centres never move.
        all_results = two_stage_mc_greedy_runner.run_experiment(
            dataset, n_init_fraction_points=n_init_fraction_points,
            synthetic_n_classes=synthetic_n_classes,
            acquisition=acquisition, reward_update=reward_update,
            alpha_ucb=alpha_ucb,
            step_size=step_size, lambda_max=lambda_max,
            run_inference=run_inference, image_pool_side=image_pool_side,
            image_data_home=image_data_home, pred_rule=pred_rule,
            **common_kwargs
        )
        return normalize_two_stage(
            all_results, dataset, "two_stage_greedy",
            acquisition=_acquisition_label(acquisition, reward_update),
            alpha_ucb=alpha_ucb,
        )

    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def save_unified_results_to_excel(rows, filename):
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)

    # Group by Init Fraction too, NOT just Method/Dataset/Budget Fraction --
    # otherwise two_stage's std would pool together its DESIGNED
    # init_fraction sweep with genuine seed-to-seed variability, while
    # submodular's std reflects ONLY seed variability (it has no
    # init_fraction dimension). Same column header ("... (std)") would then
    # mean two different things depending on the row's Method. Grouping by
    # Init Fraction too makes every group's std mean exactly one thing:
    # variability ACROSS SEEDS ONLY, for both methods alike (a submodular
    # group has one row per seed; a two_stage group has one row per seed
    # AT THAT SPECIFIC init_fraction).
    #
    # dropna=False is required here: submodular rows have Init Fraction ==
    # NaN (no such sweep) and Feedback set / two_stage rows have Feedback
    # == NaN, and pandas' groupby DROPS NaN group keys by default -- without
    # dropna=False, those rows would silently vanish from the Summary sheet.
    # "Warm Start", "Feedback" and "Num Classes" are DESIGNED settings (not
    # seed variability), so they're group keys too.
    # "Acquisition" joins the group keys for the same reason "Feedback" and
    # "Warm Start" are there: it is a DESIGNED setting, not seed noise, so
    # pooling greedy with lp_chain rows would make the (std) columns mean
    # two different things. dropna=False (below) keeps the two_stage rows,
    # whose Acquisition is NaN.
    # "Alpha UCB" is a group key for exactly the reason "Acquisition" is:
    # it is a DESIGNED setting, so pooling alpha=2 with alpha=5 rows would
    # make every (std) column mean two different things.
    group_cols = ["Method", "Feedback", "Acquisition", "Alpha UCB",
                  "Num Classes", "Warm Start", "Dataset",
                  "Budget Fraction", "Init Fraction"]
    numeric_cols = [
        c for c in UNIFIED_COLUMNS
        if c not in group_cols + ["Seed", "Selected Subsets"]
    ]
    summary = (
        df.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]} ({col[1]})" for col in summary.columns
    ]

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(filename)
    _style_sheet(wb["Detailed Results"])
    _style_sheet(wb["Summary"])
    wb.save(filename)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one proposed MULTICLASS AFA method (submodular / two_stage) on a "
                     "real AFA-Benchmark or synthetic dataset, and write results in one unified "
                     "schema."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-modalities", type=str, default=None,
                         help="Integer, or 'all' for no truncation. Controls modality/view "
                              "count for BOTH dataset kinds: for real datasets, keeps only the "
                              "first N feature columns; for synthetic datasets, GENERATES exactly "
                              "N views in the first place ('all' falls back to the synthetic "
                              "default view count since there's no dataset width to truncate). "
                              "Defaults to each method's own recommended default if omitted "
                              "(both currently: all).")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap a REAL dataset to at most this many rows (reproducible "
                              "subsample; if the dataset already has fewer rows, all of them "
                              "are used). Ignored for synthetic datasets -- use --n-samples "
                              "instead.")
    parser.add_argument("--budget-fractions", type=str, default="0,0.1,0.3,0.5,0.7,0.9,1.1")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--n-init-fraction-points", type=int, default=10,
                         help="two_stage_greedy only: number of init_fraction (gamma) points swept "
                              "per budget fraction. Ignored for submodular.")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES,
                         help="synthetic datasets only: how many rows to generate. Ignored for "
                              "real datasets.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED,
                         help="synthetic datasets only: seed for the generative means draw "
                              "(independent of --seeds, which controls the train/inference split).")
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE,
                         help="synthetic datasets only: per-(class,view) means drawn ~ "
                              "Uniform(0, mean_scale).")
    parser.add_argument("--output-xlsx", type=str, default=None,
                         help="If not given, auto-named as results/results_{method}_"
                              "{dataset}_max{N or ALL}_seeds{n_seeds}.xlsx ('ALL' when "
                              "--max-modalities all was passed; the 'results/' subdirectory is "
                              "created if missing, relative to wherever the script is run from). "
                              "If given explicitly, that path is used as-is (NOT forced under "
                              "results/).")
    parser.add_argument("--feedback", choices=("full", "bandit"), default="full",
                         help="submodular only: how the CENTRES are updated. 'full' reveals "
                              "y_true every round; 'bandit' observes only the one-bit "
                              "correct/incorrect reward. No counterpart in two_stage_greedy, "
                              "whose centres are frozen after Stage 1 -- there the full/bandit "
                              "distinction lives entirely in --reward-update. NOTE: --feedback "
                              "bandit is INCOMPATIBLE with --reward-update subsets (the replay "
                              "reads y_true, which bandit feedback does not reveal); that "
                              "combination is rejected rather than silently run.")
    parser.add_argument("--acquisition", choices=ACQUISITION_MODES, default="greedy",
                         help="BOTH methods: the per-round ACQUISITION policy, and the one "
                              "axis on which the two methods are directly comparable. "
                              "'greedy' (DEFAULT): the submodular greedy oracle. 'lp_chain': "
                              "UCB reward estimates over the nviews+1 nested greedy chain, "
                              "with the per-round budgeted LP solved exactly each round; runs "
                              "at any nviews. 'lp_full': the same over the FULL 2^(nviews-1) "
                              "enumeration -- the small-nviews fidelity check for what the "
                              "chain restriction costs, capped at MAX_REWARD_ESTIMATE_VIEWS "
                              "views. 'lp_full_opt': the ORACLE CEILING for lp_full -- same "
                              "action space, but the arm values are the EXACT accuracies of "
                              "the TRUE generative means instead of UCB estimates, and the LP "
                              "is solved ONCE before the loop, so each round is a single draw "
                              "from a frozen distribution. lp_full_opt minus lp_full is the "
                              "price of LEARNING the arm values, holding action space and LP "
                              "fixed. SYNTHETIC DATASETS ONLY (there is no true mean for real "
                              "data); --reward-update and --alpha-ucb are both inert under it. "
                              "It does NOT give the classifier the true means -- submodular "
                              "still learns centres under --feedback, two_stage_greedy still "
                              "predicts with its Stage-1 centres -- so it is a ceiling on the "
                              "acquisition axis alone. Being non-adaptive by construction (it "
                              "never looks at x_t or at surviving budget), the ADAPTIVE modes "
                              "may legitimately score above it; see core/optimal_static.py's "
                              "'IS NOT a bound for ADAPTIVE acquisition' note. "
                              "BOTH methods take these four names verbatim -- there "
                              "is no per-method translation. For two_stage_greedy all three "
                              "hold the Stage-1 centres FROZEN, which is what makes them a "
                              "clean one-axis comparison; the four learning-centre modes it "
                              "used to carry (reward_full / reward_bandit / full / bandit) "
                              "have been REMOVED -- see that module's HISTORY docstring.")
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                         help="synthetic only: how many classes to generate "
                              "(labels {0..K-1}). Every other dataset infers its class "
                              "count from the labels.")
    parser.add_argument("--step-size", type=float, default=1.0,
                         help="two_stage_greedy only: Stage-2 OMD dual (lambda) ascent step size "
                              "(default 1.0). Ignored for submodular.")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                         help="two_stage_greedy only: Stage-2 OMD dual variable clip ceiling "
                              "(default 10.0). Ignored for submodular.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                         help="BOTH methods, under --acquisition lp_chain / lp_full only: "
                              "the arm-reward update scope -- i.e. the full-vs-bandit feedback "
                              "distinction as it applies to the ACQUISITION policy. "
                              "'subsets' (DEFAULT): counterfactual replay over every arm "
                              "contained in the acquired set (full feedback, many "
                              "observations/round). 'selected': the played arm only, from its "
                              "0/1 reward (bandit feedback). Ignored otherwise.")
    parser.add_argument("--alpha-ucb", type=float, default=2.0,
                         help="BOTH methods: optimism scale in the "
                              "bonus sqrt(alpha_ucb*log(t+1))/sqrt(count) (default 2.0). 0 "
                              "disables optimism. NOTE on two_stage_greedy: its centres are "
                              "frozen and stage1_counts is identical across views, so the "
                              "bonus adds the SAME constant to every view -- it acts as a "
                              "slowly growing preference for larger view sets, not as "
                              "per-view optimism. Only on submodular, whose counts genuinely "
                              "diverge across views, does it steer WHICH view to explore.")
    parser.add_argument("--lr", type=float, default=1e-2,
                         help="submodular --feedback bandit only: the complementary-label "
                              "gradient step (default 1e-2). Ignored for two_stage_greedy, "
                              "whose centres never move.")
    parser.add_argument("--skip-inference", action="store_true",
                         help="Run ONLY Stage 1 + Stage 2 training; skip the Stage-2 LP "
                              "inference step. Supported by BOTH methods. Split and budgets are "
                              "unchanged; all inference_* columns become NaN and Total Reward "
                              "becomes the train-only figure. Auto-named output gets a "
                              "_trainonly tag.")
    parser.add_argument("--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE,
                         help=f"mnist/fashion_mnist only: block-average each 28x28 image down to "
                              f"x*x features (x=this value; default {DEFAULT_IMAGE_POOL_SIDE} -> "
                              f"{DEFAULT_IMAGE_POOL_SIDE**2} features). Pass 28 to keep all 784 "
                              f"raw pixels (intractable for two_stage). For two_stage keep "
                              f"x small (Stage 2 uses 2^(x*x-1) experts). Ignored for non-image "
                              f"datasets.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                         help="mnist/fashion_mnist only: directory fetch_openml caches its "
                              "download in. Default (None) resolves to core.datasets."
                              "DEFAULT_OPENML_DATA_HOME, a RELATIVE 'data/openml_cache' folder -- "
                              "deliberately NOT fetch_openml's own ~/scikit_learn_data default, "
                              "which exceeds $HOME's quota on clusters like NCI Gadi. Point this "
                              "at your project/scratch space if the default location itself lacks "
                              "quota. Ignored for non-image datasets.")
    parser.add_argument("--pred-rule", choices=PRED_RULES, default="nearest_center",
                         help="two_stage_greedy only: hard-decision prediction rule. "
                              "'nearest_center' (default; two_stage's original K-way argmin) or "
                              "'pairwise_vote' (submodular's one-vs-one linear-discriminant "
                              "vote). NOTE: these are mathematically EQUIVALENT given the same "
                              "observed views (pairwise vote reduces exactly to nearest-centroid), "
                              "so this does NOT change accuracy -- it exists to confirm "
                              "two_stage_greedy and submodular share a decision rule. Ignored for "
                              "submodular (which always uses pairwise-vote intrinsically).")
    args = parser.parse_args()

    # Reject the one incoherent flag combination up front, as a clean
    # argparse error rather than a traceback out of the training loop.
    # See gmm_multiclass_submodular.run_training_phase's guard for why.
    if (args.method == "submodular" and args.feedback == "bandit"
            and args.reward_update == "subsets"
            and args.acquisition in ("lp_chain", "lp_full")):
        parser.error(
            "--feedback bandit is incompatible with --reward-update subsets: the "
            "counterfactual replay reads y_true, which bandit feedback does not "
            "reveal. Use '--feedback bandit --reward-update selected' or "
            "'--feedback full --reward-update subsets'.")

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.max_modalities is None:
        max_modalities = DEFAULT_MAX_MODALITIES[args.method]
    else:
        max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    t0 = time.time()
    rows = run_method(
        args.method, args.dataset, max_modalities, seeds, budget_fractions,
        args.data_path, args.n_init_fraction_points, args.max_samples,
        args.n_samples, args.synthetic_seed, args.mean_scale,
        feedback=args.feedback,
        synthetic_n_classes=args.num_classes,
        step_size=args.step_size,
        lambda_max=args.lambda_max, run_inference=not args.skip_inference,
        image_pool_side=args.image_pool_side,
        image_data_home=args.image_cache_dir,
        pred_rule=args.pred_rule,
        reward_update=args.reward_update,
        alpha_ucb=args.alpha_ucb, lr=args.lr,
        acquisition=args.acquisition,
    )

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    print(f"\n{'=' * 70}\nSUMMARY (mean +/- std across seeds"
          f"{', averaged over init fractions' if args.method in TWO_STAGE_FAMILY else ''}"
          f") -- {args.method} / {args.dataset}"
          f" / {args.acquisition}"
          f"\n{'=' * 70}")
    print(f"{'Fraction':<10}{'Train Rew':>12}{'Train F1':>10}{'Train AUROC':>13}"
          f"{'Inf Rew':>10}{'Inf F1':>9}{'Inf AUROC':>11}{'Train Spent':>13}{'Inf Spent':>12}")
    print("-" * 100)
    for frac, sub in df.groupby("Budget Fraction"):
        print(
            f"{frac:<10.2f}"
            f"{sub['Train Reward'].mean():>9.3f} "
            f"{sub['Train F1'].mean():>8.3f} "
            f"{sub['Train AUROC'].mean():>11.3f} "
            f"{sub['Inference Reward'].mean():>8.3f} "
            f"{sub['Inference F1'].mean():>7.3f} "
            f"{sub['Inference AUROC'].mean():>9.3f} "
            f"{sub['Train Spent'].mean():>11.4f} "
            f"{sub['Inference Spent'].mean():>10.4f}"
        )

    if args.output_xlsx:
        # Explicit path given -- used as-is, NOT forced under results/.
        output_xlsx = args.output_xlsx
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        ti_tag = "_trainonly" if args.skip_inference else ""
        fb_tag = f"_{args.feedback}" if args.method == "submodular" else ""
        # BOTH methods, one rule. Empty under the default --acquisition
        # greedy, so pre-existing submodular filenames are unchanged.
        # two_stage_greedy filenames DO change shape: the old {cu_tag}
        # component is gone, so e.g. "..._frozen..." becomes "..._greedy..."
        # -- deliberate, since both methods now name the same policy the
        # same way.
        acq_tag = ""
        if args.acquisition in ORACLE_ACQUISITION_MODES:
            # No reward_update component (never scored) and no alpha component
            # (no UCB bonus exists to scale) -- both would name knobs the run
            # ignored, and would write distinct filenames for identical runs.
            acq_tag = f"_{args.acquisition}"
        elif args.acquisition != "greedy":
            acq_tag = f"_{args.acquisition}-{args.reward_update}"
            if args.alpha_ucb != 2.0:
                acq_tag += f"_alpha{args.alpha_ucb:g}"
        # pairwise_vote only tagged for two_stage (submodular always
        # uses it intrinsically, so it'd be redundant noise there) -- avoids
        # --pred-rule nearest_center vs --pred-rule pairwise_vote silently
        # overwriting the same file.
        pr_tag = ("_pairwisevote" if (args.pred_rule == "pairwise_vote"
                                       and args.method in TWO_STAGE_FAMILY) else "")
        # max_modalities is None when --max-modalities all was passed (no
        # truncation) -- render that as "ALL" in the filename instead of
        # the confusing literal "maxNone".
        maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
        # Synthetic datasets only: include the actual class count in the
        # filename (--num-classes for synthetic; fixed at 2 for
        # the binary synthetic generators) so runs at different K don't
        # overwrite each other. Real datasets aren't tagged -- their class
        # count is fixed by the data itself, not a run parameter.
        classes_tag = ""
        if args.dataset in SYNTHETIC_DATASETS:
            n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
            classes_tag = f"_classes{n_classes}"
        output_xlsx = str(
            results_dir / f"results_{args.method}{fb_tag}{acq_tag}_{args.dataset}_max{maxmod_label}_seeds{len(seeds)}{ti_tag}{pr_tag}{classes_tag}.xlsx"
        )
    save_unified_results_to_excel(rows, output_xlsx)
    print(f"Execution time: {time.time() - t0:.1f} seconds")