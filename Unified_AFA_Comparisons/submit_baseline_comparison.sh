#!/bin/bash
# Fair proposed-vs-baseline comparison.
# Running this file submits 36 jobs, all using the same 60/20/20 protocol,
# five seeds, effective feature restrictions, and full inference.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results

SEEDS="${SEEDS:-42,43,44,45,46}"
SEEDS_PBS="${SEEDS//,/:}"
PYTHON_BIN="${PYTHON_BIN:-/g/data/zf94/Supervised_AFA/env/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

submit_proposed () {
    local method="$1" dataset="$2" walltime="$3" maxmod="$4"
    local extra_args="$5" tag="$6"
    local split_mode="60-20-20"
    local jobname="${method}_vsbase_${dataset}_${tag}"
    local -a extra_argv=()
    if [ -n "${extra_args}" ]; then read -r -a extra_argv <<< "${extra_args}"; fi

    local output_stem
    output_stem=$("${PYTHON_BIN}" run_proposed_methods.py \
        --method "${method}" --dataset "${dataset}" \
        --split-mode "${split_mode}" --max-modalities "${maxmod}" \
        --seeds "${SEEDS}" "${extra_argv[@]}" --print-output-stem)

    echo "Submitting proposed ${method}/${dataset}/${tag}: ${output_stem}"
    export METHOD="${method}" DATASET="${dataset}" SPLIT_MODE="${split_mode}"
    export MAXMOD="${maxmod}" SEEDS_PBS EXTRA_ARGS="${extra_args}"
    qsub -N "${jobname}" -l walltime="${walltime}" \
        -v METHOD,DATASET,SPLIT_MODE,MAXMOD,SEEDS_PBS,EXTRA_ARGS \
        -o "logs/${output_stem}.log" run_single_proposed.pbs
}

submit_baseline () {
    local method="$1" dataset="$2" walltime="$3" maxmod="$4"
    local extra_args="$5" tag="$6"
    case "${method}" in eddi|dime) ;; *) echo "Use eddi or dime individually" >&2; exit 2 ;; esac
    local jobname="${method}_${dataset}_${tag}"
    local seed_count; seed_count=$(awk -F',' '{print NF}' <<< "${SEEDS}")
    local logname="${jobname}_max${maxmod}_seeds${seed_count}.log"

    echo "Submitting baseline ${method}/${dataset}/${tag}"
    export METHOD="${method}" DATASET="${dataset}" MAXMOD="${maxmod}"
    export SEEDS_PBS EXTRA_ARGS="${extra_args}"
    qsub -N "${jobname}" -l walltime="${walltime}" \
        -v METHOD,DATASET,MAXMOD,SEEDS_PBS,EXTRA_ARGS \
        -o "logs/${logname}" run_single_baseline.pbs
}

# Proposed methods: greedy only, empirical rewards, full inference.
# Synthetic data: 10 views, 1,000 samples, both K=2 and K=4.
submit_proposed adaptive synthetic  01:00:00  10 "--feedback full --acquisition greedy --reward-estimate empirical --num-classes 2 --n-samples 1000" "greedy_c2"
submit_proposed adaptive synthetic  01:00:00  10 "--feedback full --acquisition greedy --reward-estimate empirical --num-classes 4 --n-samples 1000" "greedy_c4"
submit_proposed two_stage synthetic 12:00:00  10 "                --acquisition greedy --reward-estimate empirical --num-classes 2 --n-samples 1000" "greedy_c2"
submit_proposed two_stage synthetic 12:00:00  10 "                --acquisition greedy --reward-estimate empirical --num-classes 4 --n-samples 1000" "greedy_c4"

# Real data: reproduce Compared_Baselines restrictions exactly.
for method in adaptive two_stage; do
    if [ "${method}" = adaptive ]; then short_wall=01:00:00; long_wall=01:00:00; method_args="--feedback full"; else short_wall=12:00:00; long_wall=24:00:00; method_args=""; fi
    submit_proposed "${method}" ckd            "${short_wall}" 10 "${method_args} --acquisition greedy --reward-estimate empirical"                                         "greedy"
    submit_proposed "${method}" actg175        "${short_wall}" 10 "${method_args} --acquisition greedy --reward-estimate empirical"                                         "greedy"
    submit_proposed "${method}" physionet      "${long_wall}"  10 "${method_args} --acquisition greedy --reward-estimate empirical"                                         "greedy"
    submit_proposed "${method}" bank_marketing "${long_wall}"  10 "${method_args} --acquisition greedy --reward-estimate empirical --max-samples 10000"                     "greedy"
    submit_proposed "${method}" diabetes       "${long_wall}"  10 "${method_args} --acquisition greedy --reward-estimate empirical --max-samples 10000"                     "greedy"
    submit_proposed "${method}" mnist          "${long_wall}"  10 "${method_args} --acquisition greedy --reward-estimate empirical --max-samples 10000 --image-pool-side 4" "greedy"
    submit_proposed "${method}" fashion_mnist  "${long_wall}"  10 "${method_args} --acquisition greedy --reward-estimate empirical --max-samples 10000 --image-pool-side 4" "greedy"
done

# EDDI and DIME are always separate jobs--never --method all.
for method in eddi dime; do
    submit_baseline "${method}" synthetic 00:30:00 10 "--n-views 10 --num-classes 2 --n-samples 1000" "c2"
    submit_baseline "${method}" synthetic 00:30:00 10 "--n-views 10 --num-classes 4 --n-samples 1000" "c4"
done

submit_baseline eddi ckd            00:15:00 10 ""                                        "real"
submit_baseline eddi actg175        00:20:00 10 ""                                        "real"
submit_baseline eddi physionet      01:45:00 10 ""                                        "real"
submit_baseline eddi bank_marketing 02:30:00 10 "--max-samples 10000"                     "real"
submit_baseline eddi diabetes       01:30:00 10 "--max-samples 10000"                     "real"
submit_baseline eddi mnist          04:15:00 10 "--max-samples 10000 --image-pool-side 4" "real"
submit_baseline eddi fashion_mnist  04:15:00 10 "--max-samples 10000 --image-pool-side 4" "real"

submit_baseline dime ckd            00:10:00 10 ""                                        "real"
submit_baseline dime actg175        00:15:00 10 ""                                        "real"
submit_baseline dime physionet      00:25:00 10 ""                                        "real"
submit_baseline dime bank_marketing 00:40:00 10 "--max-samples 10000"                     "real"
submit_baseline dime diabetes       00:40:00 10 "--max-samples 10000"                     "real"
submit_baseline dime mnist          01:30:00 10 "--max-samples 10000 --image-pool-side 4" "real"
submit_baseline dime fashion_mnist  01:30:00 10 "--max-samples 10000 --image-pool-side 4" "real"

echo "All baseline-comparison jobs submitted. Check with: qstat -u $USER"
