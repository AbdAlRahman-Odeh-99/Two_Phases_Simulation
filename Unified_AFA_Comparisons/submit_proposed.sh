#!/bin/bash
# Proposed-method comparison: mirrors Proposed_Approaches/submit_multiclass.sh.
# Running this file submits 16 training-only jobs using the 80/20 protocol.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results

SEEDS="${SEEDS:-$(seq -s ',' 42 51)}"
SEEDS_PBS="${SEEDS//,/:}"
PYTHON_BIN="${PYTHON_BIN:-/g/data/zf94/Supervised_AFA/env/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

submit () {
    local method="$1" dataset="$2" walltime="$3" maxmod="$4"
    local extra_args="$5" tag="$6"
    local split_mode="80-20"
    local jobname="${method}_${dataset}_${tag}"
    local -a extra_argv=()
    if [ -n "${extra_args}" ]; then read -r -a extra_argv <<< "${extra_args}"; fi

    local output_stem
    output_stem=$("${PYTHON_BIN}" run_proposed_methods.py \
        --method "${method}" --dataset "${dataset}" \
        --split-mode "${split_mode}" --max-modalities "${maxmod}" \
        --seeds "${SEEDS}" "${extra_argv[@]}" --print-output-stem)

    echo "Submitting ${jobname}: ${output_stem}"
    export METHOD="${method}" DATASET="${dataset}" SPLIT_MODE="${split_mode}"
    export MAXMOD="${maxmod}" SEEDS_PBS EXTRA_ARGS="${extra_args}"
    qsub -N "${jobname}" -l walltime="${walltime}" \
        -v METHOD,DATASET,SPLIT_MODE,MAXMOD,SEEDS_PBS,EXTRA_ARGS \
        -o "logs/${output_stem}.log" run_single_proposed.pbs
}

COMMON_C2="--skip-inference --step-size 1 --num-classes 2 --n-samples 1000"
COMMON_C4="--skip-inference --step-size 1 --num-classes 4 --n-samples 1000"

submit adaptive synthetic 03:00:00 15 "--acquisition hedge                                 ${COMMON_C2}" "hedge_c2"
submit adaptive synthetic 03:00:00 15 "--acquisition greedy --reward-estimate empirical    ${COMMON_C2}" "greedy_empirical_c2"
submit adaptive synthetic 03:00:00 15 "--acquisition lp_chain --reward-estimate empirical  ${COMMON_C2}" "lpchain_empirical_c2"
submit adaptive synthetic 03:00:00 15 "--acquisition lp_full_opt                           ${COMMON_C2}" "lpfullopt_c2"

submit adaptive synthetic 03:00:00 15 "--acquisition hedge                                 ${COMMON_C4}" "hedge_c4"
submit adaptive synthetic 03:00:00 15 "--acquisition greedy --reward-estimate empirical    ${COMMON_C4}" "greedy_empirical_c4"
submit adaptive synthetic 03:00:00 15 "--acquisition lp_chain --reward-estimate empirical  ${COMMON_C4}" "lpchain_empirical_c4"
submit adaptive synthetic 03:00:00 15 "--acquisition lp_full_opt                           ${COMMON_C4}" "lpfullopt_c4"

submit two_stage synthetic 24:00:00 15 "--acquisition hedge                                ${COMMON_C2}" "hedge_c2"
submit two_stage synthetic 24:00:00 15 "--acquisition greedy --reward-estimate empirical   ${COMMON_C2}" "greedy_empirical_c2"
submit two_stage synthetic 24:00:00 15 "--acquisition lp_chain --reward-estimate empirical ${COMMON_C2}" "lpchain_empirical_c2"
submit two_stage synthetic 24:00:00 15 "--acquisition lp_full_opt                          ${COMMON_C2}" "lpfullopt_c2"

submit two_stage synthetic 24:00:00 15 "--acquisition hedge                                ${COMMON_C4}" "hedge_c4"
submit two_stage synthetic 24:00:00 15 "--acquisition greedy --reward-estimate empirical   ${COMMON_C4}" "greedy_empirical_c4"
submit two_stage synthetic 24:00:00 15 "--acquisition lp_chain --reward-estimate empirical ${COMMON_C4}" "lpchain_empirical_c4"
submit two_stage synthetic 24:00:00 15 "--acquisition lp_full_opt                          ${COMMON_C4}" "lpfullopt_c4"

echo "All proposed-comparison jobs submitted. Check with: qstat -u $USER"
