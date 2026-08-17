"""
run_proposed_methods.py

Single CLI entry point that runs any ONE of the proposed MULTICLASS AFA
methods -- each of which works on any registered dataset (the real
AFA-Benchmark datasets, where nclasses is inferred from the labels, or the
synthetic GMM datasets including synthetic) --

    --method adaptive       -> gmm_multiclass_adaptive_runner.run_experiment
    --method two_stage -> two_stage_multiclass_greedy_runner.run_experiment

-- and writes ONE Excel file with a unified row schema across both, so
results from different methods can be concatenated/compared directly
instead of living in differently-shaped workbooks.

=== Usage ===
    python run_proposed_methods.py --method adaptive --dataset synthetic --num-classes 4
    python run_proposed_methods.py --method adaptive --dataset ckd --feedback bandit
    python run_proposed_methods.py --method adaptive --dataset synthetic --acquisition lp_chain
    python run_proposed_methods.py --method adaptive --dataset synthetic --acquisition lp_full --reward-update selected --max-modalities 8
    python run_proposed_methods.py --method two_stage --dataset synthetic --num-classes 3 --max-modalities 10
    python run_proposed_methods.py --method two_stage --dataset actg175 --acquisition lp_chain
    # oracle ceiling for lp_full (synthetic only -- it needs the generative means):
    python run_proposed_methods.py --method adaptive --dataset synthetic --acquisition lp_full_opt --max-modalities 8
    python run_proposed_methods.py --method adaptive --dataset synthetic --acquisition ucb_argmax --max-modalities 8
    python run_proposed_methods.py --method two_stage --dataset synthetic --acquisition ucb_argmax --max-modalities 8
    # salvage a workbook from a job that was killed mid-sweep:
    python run_proposed_methods.py --rebuild-from results/<run_id>.rows.jsonl

Run `python run_proposed_methods.py --help` for the full flag list.

=== Observability ===
Every run now has an identity (run_id) shared by four artefacts:

    logs/{run_id}.log             console + DEBUG detail, timestamped
    results/{run_id}.manifest.json  git commit + dirty flag, argv, resolved
                                    args, package versions, hostname, PBS
                                    job id, timing, failures
    results/{run_id}.rows.jsonl   every row, appended and flushed as it is
                                    produced -- the crash checkpoint
    results/{...}.xlsx            the workbook, unchanged in shape apart
                                    from appended columns and a Run Info
                                    sheet mirroring the manifest

See core/logging_utils.py for the rationale behind each. The run_id is
derived from the output workbook's stem, so all four sort together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from core.excel_utils import serialize_selected_subsets, style_and_save
from core.logging_utils import (
    EXCEL_TIMING_COLUMNS,
    TIMING_COLUMNS,
    excel_timing_columns,
    get_logger,
    read_manifest,
    read_rows_jsonl,
    setup_run,
)

import gmm_submodular.gmm_multiclass_submodular_runner as multiclass_runner
import two_stage_greedy.two_stage_multiclass_greedy_runner as two_stage_mc_greedy_runner
from core.multiclass_common import PRED_RULES

# --acquisition and --reward-update are SHARED by both methods and mean the
# same thing in each, so both read from ONE definition in core rather than
# from per-method copies the driver would have to assert were in sync.
from core.submodular_greedy import (
    ACQUISITION_MODES,
    ARGMAX_ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
    REWARD_ESTIMATES,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
    REWARD_UPDATE_SCOPES,
)

log = get_logger("afa.driver")


METHODS = ("adaptive", "two_stage")
TWO_STAGE_FAMILY = ("two_stage",)

# FILENAME LABELS ONLY -- not the --method vocabulary, not the "Method"
# column in the workbook. "two_stage" names the METHOD (two-stage,
# with the greedy-family Stage 2 that replaced EXP4), but in a filename it
# sits right next to the ACQUISITION tag, where it reads as though the run
# used --acquisition greedy:
#     results_two_stage_lp_chain-subsets_...   <- greedy or lp_chain?
# The acquisition is now always tagged (see the filename block in __main__),
# so the method half only has to identify the family:
#     results_two_stage_lp_chain-subsets_...
# --method still takes "two_stage", and the "Method" column still
# reads "two_stage", so nothing that groups or filters on either
# changes -- this is a rename of the default output PATH only.
FILENAME_METHOD_LABELS = {"two_stage": "two_stage"}
DEFAULT_MAX_MODALITIES = {"adaptive": None, "two_stage": None}


def _nan(d, key):
    """d[key] if present, else NaN.

    Used throughout the normalizers below. Before per-cell failure
    isolation existed, every row was complete and direct indexing was safe;
    now a failed cell contributes a row carrying only its coordinates and
    its status, and a KeyError there would defeat the entire point of
    surviving the failure.
    """
    v = d.get(key, np.nan)
    return np.nan if v is None else v


# ─────────────────────────────────────────────────────────────────────────
# Normalization: reshape each method's native results into one common
# row schema (a list of flat dicts), ready for pd.DataFrame(rows).
# ─────────────────────────────────────────────────────────────────────────
def _timing_fragment(source, indexer=None):
    """{Excel column name: value} for the timing block.

    `source` is either a flat row dict (two_stage) or the per-fraction
    dict-of-lists (adaptive), in which case `indexer` is the seed index.
    """
    out = {}
    for key in TIMING_COLUMNS:
        col = EXCEL_TIMING_COLUMNS[key]
        if indexer is None:
            out[col] = _nan(source, key)
        else:
            series = source.get(key)
            out[col] = (series[indexer]
                        if series is not None and indexer < len(series) else np.nan)
    return out


def _normalize_frac_keyed_results(results, budget_fractions, seeds, dataset_name, method_name,
                                   feedback=np.nan, n_classes=2,
                                   acquisition=np.nan, reward_estimate=np.nan,
                                    alpha_ucb=np.nan):
    """Normalizer for adaptive's results dict, which has the shape
    {budget_fraction: {metric_name: [per-seed values]}}.

    feedback / n_classes / acquisition / alpha_ucb: experiment-level
    settings recorded per row. alpha_ucb is taken from the caller because
    gmm_multiclass_adaptive_runner does not put it in its results dict
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
                "Reward Estimate": reward_estimate,
                "Alpha UCB": alpha_ucb,
                "Num Classes": n_classes,
                "Warm Start": False,  # adaptive has no warm-start concept
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
                "Status": (r["status"][i] if "status" in r else "ok"),
                "Error": (r["error_msg"][i] if "error_msg" in r else ""),
            }
            row.update(_timing_fragment(r, i))
            rows.append(row)
    return rows


def normalize_two_stage(all_results, dataset_name,
                        method_name="two_stage", acquisition=np.nan,
                        reward_estimate=np.nan, alpha_ucb=np.nan):
    """Normalizer for two_stage_multiclass_runner's flat list-of-dicts
    results. Num Classes is read from each row's own 'nclasses' (inferred
    from the labels, or --num-classes for synthetic), NOT
    hardcoded.

    alpha_ucb: fallback for the 'Alpha UCB' column; each row's own
    'alpha_ucb' is preferred when the runner recorded one, so a caller
    that sweeps alpha per row is reported correctly.

    acquisition: the run's acquisition policy as one string, for the
    shared 'Acquisition' column -- two_stage passes
the same "greedy" / "lp_chain+subsets" / "lp_full+selected" strings
    adaptive reports, since both methods now share the axis.

    Every field is read through _nan(), so a status="error" row -- which
    carries its coordinates and nothing else -- normalizes to a row of NaNs
    instead of raising."""
    rows = []
    for d in all_results:
        row = {
            "Method": method_name,
            "Feedback": np.nan,
            "Acquisition": acquisition,
            "Reward Estimate": d.get("reward_estimate", reward_estimate),
            "Alpha UCB": d.get("alpha_ucb", alpha_ucb),
            "Num Classes": _nan(d, "nclasses"),
            "Warm Start": bool(d.get("warm_start", False)),
            "Dataset": dataset_name,
            "Seed": _nan(d, "seed"),
            "Budget Fraction": _nan(d, "budget_fraction"),
            "Init Fraction": _nan(d, "init_fraction"),
            "Train Reward": _nan(d, "avg_reward"),
            "Train F1": _nan(d, "train_f1"),
            "Train AUROC": _nan(d, "train_auroc"),
            "Inference Reward": _nan(d, "inference_accuracy"),
            "Inference F1": _nan(d, "inference_f1"),
            "Inference AUROC": _nan(d, "inference_auroc"),
            "Total Reward": _nan(d, "total_reward"),
            "Two Stage Error": _nan(d, "two_stage_error"),
            "Init Error": _nan(d, "init_error"),
            "Train Spent": _nan(d, "training_budget_spent"),
            "Inference Spent": _nan(d, "inference_actual_cost"),
            "Train Time (s)": _nan(d, "train_time_sec"),
            "Inference Time (s)": _nan(d, "inference_time_sec"),
            "Seed Time (s)": _nan(d, "seed_time_sec"),
            "Num Masks Inference": _nan(d, "num_masks_inference"),
            "Train Samples": _nan(d, "n_train"),
            "Inference Samples": _nan(d, "n_test"),
            "Train Budget": _nan(d, "training_budget"),
            "Inference Budget": _nan(d, "inference_budget"),
            "Num Arms": _nan(d, "n_arms"),
            "Selected Subsets": d.get(
                "Selected Subsets",
                serialize_selected_subsets(d.get("selected_subsets", [])),
            ),
            "Status": d.get("status", "ok"),
            "Error": d.get("error_msg", ""),
        }
        row.update(_timing_fragment(d))
        rows.append(row)
    return rows


def normalize_adaptive_flat_rows(rows, dataset_name, feedback=np.nan, n_classes=np.nan,
                                 acquisition=np.nan, reward_estimate=np.nan,
                                 alpha_ucb=np.nan):
    """Normalizer for adaptive rows read back from a .rows.jsonl checkpoint.

    The checkpoint stores adaptive's cells FLAT (one dict per cell with its
    seed and budget fraction) rather than in the dict-of-lists shape
    run_experiment returns, because a checkpoint is written one cell at a
    time -- that is the whole point of it. So rebuilding needs this
    counterpart to _normalize_frac_keyed_results. Used only by
    --rebuild-from.
    """
    out = []
    for d in rows:
        row = {
            "Method": "adaptive",
            "Feedback": feedback,
            "Acquisition": acquisition,
            "Reward Estimate": reward_estimate,
            "Alpha UCB": alpha_ucb,
            "Num Classes": n_classes,
            "Warm Start": False,
            "Dataset": dataset_name,
            "Seed": _nan(d, "seed"),
            "Budget Fraction": _nan(d, "budget_fraction"),
            "Init Fraction": np.nan,
            "Train Reward": _nan(d, "train_reward"),
            "Train F1": _nan(d, "train_f1"),
            "Train AUROC": _nan(d, "train_auroc"),
            "Inference Reward": _nan(d, "inference_reward"),
            "Inference F1": _nan(d, "inference_f1"),
            "Inference AUROC": _nan(d, "inference_auroc"),
            "Total Reward": _nan(d, "total_reward"),
            "Two Stage Error": np.nan,
            "Init Error": np.nan,
            "Train Spent": _nan(d, "train_spent"),
            "Inference Spent": _nan(d, "inference_spent"),
            "Train Time (s)": _nan(d, "train_time_sec"),
            "Inference Time (s)": _nan(d, "inference_time_sec"),
            # seed_time_sec is only known once a seed's whole budget loop
            # finishes, so a checkpoint written mid-seed does not have it.
            "Seed Time (s)": _nan(d, "seed_time_sec"),
            "Num Masks Inference": _nan(d, "num_masks_inference"),
            "Train Samples": _nan(d, "n_train"),
            "Inference Samples": _nan(d, "n_inference"),
            "Train Budget": _nan(d, "train_budget"),
            "Inference Budget": _nan(d, "inference_budget"),
            "Num Arms": _nan(d, "n_arms"),
            "Selected Subsets": "",   # not checkpointed -- see emit_row
            "Status": d.get("status", "ok"),
            "Error": d.get("error_msg", ""),
        }
        row.update(_timing_fragment(d))
        out.append(row)
    return out


#: The unified schema. The first block is unchanged from before -- same
#: names, same order -- and the observability columns are APPENDED, so an
#: older workbook and a newer one still concatenate (pandas fills the
#: missing new columns with NaN).
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
    "Acquisition", "Reward Estimate", "Num Arms",
    "Alpha UCB",
    "Selected Subsets",
] + excel_timing_columns() + ["Status", "Error"]

#: Columns that must never enter the numeric aggregation on the Summary
#: sheet, beyond the grouping keys. Named once so save_unified_results_to_excel
#: cannot drift from the column list above.
NON_NUMERIC_COLUMNS = ["Seed", "Selected Subsets", "Status", "Error"]


def _acquisition_label(acquisition, reward_update, reward_estimate="surrogate"):
    uses_reward_update = _uses_empirical_arm_rewards(acquisition, reward_estimate)
    if not uses_reward_update:
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
                reward_update="subsets", reward_estimate="surrogate",
                alpha_ucb=2.0, lr=1e-2,
                acquisition="greedy"):

    synthetic_n_views = max_modalities if max_modalities is not None else SYNTHETIC_N_VIEWS

    common_kwargs = dict(
        max_modalities=max_modalities, seeds=seeds, budget_fractions=budget_fractions,
        data_path=data_path, max_samples=max_samples,
        synthetic_n_samples=synthetic_n_samples, synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed, synthetic_mean_scale=synthetic_mean_scale,
    )

    uses_empirical_arm_rewards = _uses_empirical_arm_rewards(acquisition, reward_estimate)
    if (acquisition in ORACLE_ACQUISITION_MODES and reward_update != "subsets"):
        log.warning(
            "--reward-update has no effect under --acquisition %s (its arm values "
            "come from the true means and are never scored); ignored.", acquisition)
    elif (not uses_empirical_arm_rewards and reward_update != "subsets"):
        log.warning(
            "--reward-update has no effect for this acquisition/reward-estimate "
            "combination; ignored.")
    if acquisition in ORACLE_ACQUISITION_MODES and method == "adaptive" and feedback != "full":
        log.info("--acquisition %s gives the ACQUISITION policy the true means; the "
                 "classifier still learns under --feedback %s. This is not an "
                 "oracle-classifier run.", acquisition, feedback)
    if method != "adaptive" and feedback != "full":
        log.warning("--feedback governs the CENTRE update, which only adaptive has "
                    "(two_stage holds its Stage-1 centres frozen); ignored for %r. "
                    "The arm-scoring analogue is --reward-update subsets/selected.",
                    method)
    if method != "adaptive" and lr != 1e-2:
        log.warning("--lr is the complementary-label step in adaptive's --feedback "
                    "bandit centre update; ignored for %r, whose centres are frozen.",
                    method)
    if pred_rule != "nearest_center":
        log.warning("--pred-rule only applies to --method two_stage; ignored for %r.",
                    method)

    if method == "adaptive":
        results = multiclass_runner.run_experiment(
            dataset, feedback=feedback,
            acquisition=acquisition, reward_update=reward_update,
            reward_estimate=reward_estimate, alpha_ucb=alpha_ucb, lr=lr,
            step_size=step_size, lambda_max=lambda_max,
            synthetic_n_classes=synthetic_n_classes,
            run_inference=run_inference, image_pool_side=image_pool_side,
            image_data_home=image_data_home,
            **common_kwargs
        )

        if dataset in MULTICLASS_SYNTHETIC_DATASETS:
            n_classes = synthetic_n_classes
        else:
            n_classes = DATASET_N_CLASSES.get(dataset, 2)
        return _normalize_frac_keyed_results(
            results, budget_fractions, seeds, dataset, "adaptive",
            feedback=feedback, n_classes=n_classes,
            acquisition=_acquisition_label(acquisition, reward_update, reward_estimate),
            reward_estimate=reward_estimate,
            alpha_ucb=alpha_ucb,
        )

    if method == "two_stage":
        all_results = two_stage_mc_greedy_runner.run_experiment(
            dataset, n_init_fraction_points=n_init_fraction_points,
            synthetic_n_classes=synthetic_n_classes,
            acquisition=acquisition, reward_update=reward_update,
            reward_estimate=reward_estimate,
            alpha_ucb=alpha_ucb,
            step_size=step_size, lambda_max=lambda_max,
            run_inference=run_inference, image_pool_side=image_pool_side,
            image_data_home=image_data_home, pred_rule=pred_rule,
            **common_kwargs
        )
        return normalize_two_stage(
            all_results, dataset, "two_stage",
            acquisition=_acquisition_label(acquisition, reward_update, reward_estimate),
            reward_estimate=reward_estimate,
            alpha_ucb=alpha_ucb,
        )

    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def save_unified_results_to_excel(rows, filename, info_rows=None):
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    group_cols = ["Method", "Feedback", "Acquisition", "Reward Estimate", "Alpha UCB",
                  "Num Classes", "Warm Start", "Dataset",
                  "Budget Fraction", "Init Fraction"]
    numeric_cols = [
        c for c in UNIFIED_COLUMNS
        if c not in group_cols + NON_NUMERIC_COLUMNS
    ]
    # Aggregate over SUCCESSFUL cells only. An isolated failure now
    # contributes a NaN row rather than killing the sweep, and mean/std
    # over a group containing it would otherwise be NaN for every metric --
    # which would turn one lost cell back into one lost group.
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    summary = (
        ok.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]} ({col[1]})" for col in summary.columns
    ]

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    style_and_save(filename, ["Detailed Results", "Summary"], info_rows=info_rows)
    log.info("Results saved to %s", filename)


# ─────────────────────────────────────────────────────────────────────────
# Salvage: rebuild a workbook from a killed run's checkpoint
# ─────────────────────────────────────────────────────────────────────────
def rebuild_from_checkpoint(source, output_xlsx=None):
    """Turn results/{run_id}.rows.jsonl back into a workbook.

    This is what the row checkpoint is FOR. A job killed at seed 8 of 10
    used to leave nothing; now it leaves every completed cell on disk, and
    this reads them back, normalizes them through exactly the same
    functions a live run uses, and writes the same two sheets.

    The run's arguments come from the sibling manifest, so the Method /
    Dataset / Acquisition columns and the output filename are the real ones
    rather than guesses. The output gets a _partial tag: these rows are a
    prefix of the intended sweep, and a file that does not say so will
    eventually be compared against a complete one as though it were.
    """
    p = Path(source)
    if p.is_dir():
        raise ValueError(f"{source!r} is a directory; pass the .rows.jsonl file "
                         f"or the run_id")
    if not p.exists():
        # Accept a bare run_id as well as a path.
        cand = Path("results") / f"{source}.rows.jsonl"
        if not cand.exists():
            raise FileNotFoundError(f"no checkpoint at {p} or {cand}")
        p = cand

    manifest_path = Path(str(p).replace(".rows.jsonl", ".manifest.json"))
    manifest = read_manifest(manifest_path) if manifest_path.exists() else {}
    if not manifest:
        log.warning("no manifest beside %s -- rebuilding with unknown run settings; "
                    "the Method/Dataset/Acquisition columns will be blank", p)
    a = manifest.get("args", {}) or {}

    rows_native = read_rows_jsonl(p)
    if not rows_native:
        raise ValueError(f"{p} contains no readable rows")
    log.info("rebuilding from %s: %d rows, entry_point=%s",
             p, len(rows_native), manifest.get("entry_point"))

    method = a.get("method")
    if method is None:
        # A standalone runner's checkpoint has no --method; infer it from
        # which entry point wrote the manifest.
        method = "two_stage" if "two_stage" in str(manifest.get("entry_point", "")) \
            else "adaptive"
    dataset = a.get("dataset", "")
    acquisition = _acquisition_label(a.get("acquisition", "greedy"),
                                     a.get("reward_update", "subsets"),
                                     a.get("reward_estimate", "surrogate"))

    if method == "two_stage":
        rows = normalize_two_stage(
            rows_native, dataset, "two_stage", acquisition=acquisition,
            reward_estimate=a.get("reward_estimate", np.nan),
            alpha_ucb=a.get("alpha_ucb", np.nan))
    else:
        if dataset in MULTICLASS_SYNTHETIC_DATASETS:
            n_classes = a.get("num_classes", np.nan)
        else:
            n_classes = DATASET_N_CLASSES.get(dataset, np.nan)
        rows = normalize_adaptive_flat_rows(
            rows_native, dataset, feedback=a.get("feedback", np.nan),
            n_classes=n_classes, acquisition=acquisition,
            reward_estimate=a.get("reward_estimate", np.nan),
            alpha_ucb=a.get("alpha_ucb", np.nan))

    if output_xlsx is None:
        stem = a.get("output_xlsx") or manifest.get("output_xlsx")
        if stem:
            stem = Path(stem)
            output_xlsx = str(stem.with_name(stem.stem + "_partial" + stem.suffix))
        else:
            output_xlsx = str(p).replace(".rows.jsonl", "_partial.xlsx")

    info = [(k, v) for k, v in manifest.items()
            if not isinstance(v, (dict, list))]
    info.append(("rebuilt_from", str(p)))
    info.append(("rebuilt_rows", len(rows)))
    save_unified_results_to_excel(rows, output_xlsx, info_rows=info)
    return output_xlsx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one proposed MULTICLASS AFA method (adaptive / two_stage) on a "
                     "real AFA-Benchmark or synthetic dataset, and write results in one unified "
                     "schema."
    )
    parser.add_argument("--method", choices=METHODS,
                        help="Required for a normal run; omitted only with --rebuild-from.")
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
                         help="two_stage only: number of init_fraction (gamma) points swept "
                              "per budget fraction. Ignored for adaptive.")
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
                         help="adaptive only: how the CENTRES are updated. 'full' reveals "
                              "y_true every round; 'bandit' observes only the one-bit "
                              "correct/incorrect reward. No counterpart in two_stage, "
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
                              "It does NOT give the classifier the true means -- adaptive "
                              "still learns centres under --feedback, two_stage still "
                              "predicts with its Stage-1 centres -- so it is a ceiling on the "
                              "acquisition axis alone. Being non-adaptive by construction (it "
                              "never looks at x_t or at surviving budget), the ADAPTIVE modes "
                              "may legitimately score above it; see core/optimal_static.py's "
                              "'IS NOT a bound for ADAPTIVE acquisition' note. "
                              "BOTH methods take these four names verbatim -- there "
                              "is no per-method translation. For two_stage all three "
                              "hold the Stage-1 centres FROZEN, which is what makes them a "
                              "clean one-axis comparison; the four learning-centre modes it "
                              "used to carry (reward_full / reward_bandit / full / bandit) "
                              "have been REMOVED -- see that module's HISTORY docstring. "
                              "'ucb_argmax': the NOTEBOOK rule "
                              "(multiclass_supervised_unbiased_adaptive.ipynb's "
                              "sim_unbiased). Same FULL 2^(nviews-1) arm table and same "
                              "empirical UCB estimates as lp_full, but the per-round policy "
                              "is a DETERMINISTIC Lagrangian argmax -- argmax_S r_hat[S] - "
                              "lambda*cost(S) + bonus[S] -- with greedy's OMD dual in place "
                              "of the LP's shadow price, so it commits to a single subset "
                              "each round instead of sampling from a mixture. ucb_argmax "
                              "minus lp_full is therefore the price of that commitment, "
                              "holding action space and reward table fixed, just as "
                              "lp_full_opt minus lp_full is the price of learning the arm "
                              "values. --reward-update is live for it (the containment "
                              "replay is what fills the table); --reward-estimate is inert, "
                              "since the empirical table is unconditional. Capped at "
                              "MAX_REWARD_ESTIMATE_VIEWS views like lp_full, and works on "
                              "REAL data (nothing oracle about it)."
                              "'hedge': the same full empirical arm table and UCB estimates "
                              "as ucb_argmax, but selects the deterministic arm maximizing "
                              "UCB(S) / (y_time + y_cost*cost(S)), where the resource weights "
                              "are updated multiplicatively. It uses no OMD lambda. "
                              "--reward-update is live, --reward-estimate is inert, and the "
                              "same MAX_REWARD_ESTIMATE_VIEWS cap applies.")
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
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                         help="synthetic only: how many classes to generate "
                              "(labels {0..K-1}). Every other dataset infers its class "
                              "count from the labels.")
    parser.add_argument("--step-size", type=float, default=1.0,
                         help="Adaptive and two_stage: OMD dual ascent step size for "
                              "greedy and ucb_argmax. Ignored by LP acquisition modes.")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                         help="Adaptive and two_stage: upper bound for the OMD dual "
                              "variable under greedy and ucb_argmax. Ignored by LP modes.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                         help="BOTH methods: how empirical arm rewards are updated. "
                              "'subsets' replays every arm contained in the acquired "
                              "subset; 'selected' updates only the played arm from its "
                              "0/1 reward. Used by lp_chain/lp_full/ucb_argmax/hedge and by "
                              "greedy when --reward-estimate empirical. Ignored by "
                              "surrogate greedy and lp_full_opt.")
    parser.add_argument("--alpha-ucb", type=float, default=2.0,
                         help="BOTH methods: optimism scale in the "
                              "bonus sqrt(alpha_ucb*log(t+1))/sqrt(count) (default 2.0). 0 "
                              "disables optimism. NOTE on two_stage: its centres are "
                              "frozen and stage1_counts is identical across views, so the "
                              "bonus adds the SAME constant to every view -- it acts as a "
                              "slowly growing preference for larger view sets, not as "
                              "per-view optimism. Only on adaptive, whose counts genuinely "
                              "diverge across views, does it steer WHICH view to explore.")
    parser.add_argument("--lr", type=float, default=1e-2,
                         help="adaptive --feedback bandit only: the complementary-label "
                              "gradient step (default 1e-2). Ignored for two_stage, "
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
                         help="two_stage only: hard-decision prediction rule. "
                              "'nearest_center' (default; two_stage's original K-way argmin) or "
                              "'pairwise_vote' (adaptive's one-vs-one linear-discriminant "
                              "vote). NOTE: these are mathematically EQUIVALENT given the same "
                              "observed views (pairwise vote reduces exactly to nearest-centroid), "
                              "so this does NOT change accuracy -- it exists to confirm "
                              "two_stage and adaptive share a decision rule. Ignored for "
                              "adaptive (which always uses pairwise-vote intrinsically).")
    # ── observability flags (see core/logging_utils.py) ──
    parser.add_argument("--log-level", default="INFO",
                         choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                         help="CONSOLE verbosity. logs/{run_id}.log always keeps DEBUG, so "
                              "this only decides what competes for attention in the PBS .o "
                              "file; nothing is lost by lowering it.")
    parser.add_argument("--log-dir", default="logs",
                         help="Directory for {run_id}.log (default: logs/, created if "
                              "missing). On PBS point this at the same directory as the "
                              "#PBS -o path so a job's two logs sit together.")
    parser.add_argument("--trace-rounds", action="store_true",
                         help="Record one row PER TRAINING ROUND (subset, cost, lambda, "
                              "reward, remaining budget) to results/{run_id}.trace.jsonl, "
                              "and drop the 'Selected Subsets' Excel cell, which is a "
                              "strictly poorer encoding of the same thing. Off by default: "
                              "it is O(n_train) records per sweep cell.")
    parser.add_argument("--no-fine-timers", action="store_true",
                         help="Disable the fine-grained timing buckets; the t_* columns "
                              "become NaN. The original Train/Inference/Seed timings are "
                              "unaffected either way.")
    parser.add_argument("--rebuild-from", type=str, default=None,
                         help="Skip the sweep entirely: rebuild a workbook from a previous "
                              "run's results/{run_id}.rows.jsonl checkpoint (pass the path "
                              "or just the run_id). Use this after a walltime kill or an OOM "
                              "to recover every cell that completed. The output is tagged "
                              "_partial, because it is a prefix of the intended sweep.")
    args = parser.parse_args()

    if args.rebuild_from:
        # No sweep, no run context -- this is a pure file-to-file operation
        # and should not create a new run_id or a new log.
        import logging
        logging.basicConfig(level=getattr(logging, args.log_level),
                            format="%(levelname).1s %(message)s")
        out = rebuild_from_checkpoint(args.rebuild_from, args.output_xlsx)
        log.info("rebuilt %s", out)
        sys.exit(0)

    if args.method is None:
        parser.error("--method is required (omit it only with --rebuild-from)")

    uses_empirical_arm_rewards = _uses_empirical_arm_rewards(args.acquisition, args.reward_estimate)
    if (args.method == "adaptive" and args.feedback == "bandit" and args.reward_update == "subsets" and uses_empirical_arm_rewards):
        parser.error(
            "--feedback bandit is incompatible with --reward-update subsets "
            "when empirical arm rewards are used, because counterfactual "
            "replay requires y_true."
        )
    if (args.reward_estimate == "empirical" and args.acquisition not in ("greedy", "lp_chain")):
        parser.error(
            "--reward-estimate empirical applies only to "
            "--acquisition greedy or lp_chain. lp_full, ucb_argmax, and hedge "
            "already score arms from the empirical accuracy table "
            "unconditionally, so leave --reward-estimate at its default "
            "('surrogate') there."
        )

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.max_modalities is None:
        max_modalities = DEFAULT_MAX_MODALITIES[args.method]
    else:
        max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    # MOVED ABOVE the sweep. The output path is a pure function of the
    # arguments, so resolving it first costs nothing and buys two things: an
    # unwritable --output-xlsx fails in the first second rather than after
    # ten hours of compute, and the run_id can be derived from its stem so
    # the log, the manifest, the checkpoint and the workbook share a name.
    if args.output_xlsx:
        output_xlsx = args.output_xlsx
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        ti_tag = "_trainonly" if args.skip_inference else ""
        fb_tag = f"_{args.feedback}" if args.method == "adaptive" else ""
        acq_tag = f"_{args.acquisition}"
        #if args.acquisition in ("greedy", "lp_chain", "lp_full", "ucb_argmax"):
        if uses_empirical_arm_rewards:
            acq_tag += f"-{args.reward_update}"
        alpha_tag = (f"_alpha{args.alpha_ucb:g}" if (args.alpha_ucb != 2.0 and args.acquisition not in ORACLE_ACQUISITION_MODES) else "")
        reward_estimate_is_live = (args.acquisition in ("greedy", "lp_chain"))
        re_tag = (f"_{args.reward_estimate}" if reward_estimate_is_live else "")
        dual_tag = f"_step{args.step_size:g}_lmax{args.lambda_max:g}"
        #uses_omd_dual = (args.acquisition in ("greedy",) + ARGMAX_ACQUISITION_MODES)
        #dual_tag = (f"_step{args.step_size:g}_lmax{args.lambda_max:g}" if uses_omd_dual else "")
        pr_tag = ("_pairwisevote" if (args.pred_rule == "pairwise_vote" and args.method in TWO_STAGE_FAMILY) else "")
        maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
        classes_tag = ""
        if args.dataset in SYNTHETIC_DATASETS:
            n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
            classes_tag = f"_classes{n_classes}"
        output_xlsx = str(results_dir / (f"results_{FILENAME_METHOD_LABELS.get(args.method, args.method)}{fb_tag}{acq_tag}{re_tag}{alpha_tag}{dual_tag}_{args.dataset}_max{maxmod_label}_seeds{len(seeds)}{ti_tag}{pr_tag}{classes_tag}.xlsx"))

    run = setup_run(
        "run_proposed_methods",
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
    try:
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
            reward_estimate=args.reward_estimate,
            alpha_ucb=args.alpha_ucb, lr=args.lr,
            acquisition=args.acquisition,
        )
    except BaseException as exc:                          # noqa: BLE001
        # Includes the PBS walltime kill and Ctrl-C. The manifest records
        # how it ended and where the surviving rows are; recover them with
        #     python run_proposed_methods.py --rebuild-from <run_id>
        log.exception("run aborted -- recover completed cells with "
                      "--rebuild-from %s", run.rows_path)
        run.finalize(status=f"failed: {type(exc).__name__}: {exc}")
        raise

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    log.info("=" * 70)
    log.info("SUMMARY (mean +/- std across seeds%s) -- %s / %s / %s",
             ", averaged over init fractions" if args.method in TWO_STAGE_FAMILY else "",
             args.method, args.dataset, args.acquisition)
    log.info("=" * 70)
    log.info("%-10s%12s%10s%13s%10s%9s%11s%13s%12s", 'Fraction', 'Train Rew',
             'Train F1', 'Train AUROC', 'Inf Rew', 'Inf F1', 'Inf AUROC',
             'Train Spent', 'Inf Spent')
    log.info("-" * 100)
    for frac, sub in ok.groupby("Budget Fraction"):
        log.info(
            "%-10.2f%9.3f %8.3f %11.3f %8.3f %7.3f %9.3f %11.4f %10.4f",
            frac,
            sub['Train Reward'].mean(), sub['Train F1'].mean(),
            sub['Train AUROC'].mean(), sub['Inference Reward'].mean(),
            sub['Inference F1'].mean(), sub['Inference AUROC'].mean(),
            sub['Train Spent'].mean(), sub['Inference Spent'].mean(),
        )

    save_unified_results_to_excel(rows, output_xlsx, info_rows=run.info_rows())
    log.info("Execution time: %.1f seconds", time.time() - t0)
    run.finalize(status="ok")