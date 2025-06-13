#!/bin/bash
#SBATCH --job-name=solve
#SBATCH --output=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=0

. ./config/.env

MODEL_LIST=(
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen2.5-72B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.1-70B-Instruct
)

TASK_LIST=(
    # tabmwp
    # finqa
    # commonsense_qa
    # gsm8k
    # gsm_symbolic
    # gsm_symbolic_p1
    # gsm_symbolic_p2
    math_algebra
    math_counting_and_probability
    math_geometry
    math_intermediate_algebra
    math_number_theory
    math_prealgebra
    math_precalculus
    # mmlu_pro
)

ROUTE_LIST=(
    PL
    NL
)

NUM_LIST=(
    00
    01
    02
    03
)

MODEL=$1
PORT="${2:-8000}"
OTHER_ARGS=$3
PP_SIZE="${4:-1}"
# Use PP_SIZE 2 for >32B Models
TP_SIZE="${5:-1}"
# Use TP_SIZE 4 for >32B Models
ROUTE_LIST=("${6:-${ROUTE_LIST[@]}}")
TASK_LIST=("${7:-${TASK_LIST[@]}}")
NUM_LIST=("${8:-${NUM_LIST[@]}}")

MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --enable_prefix_caching \
    --guided-decoding-backend=lm-format-enforcer \
    --distributed-executor-backend mp > o.txt &

for ROUTE in "${ROUTE_LIST[@]}"
do
    for TASK in "${TASK_LIST[@]}"
    do
        for NUM in "${NUM_LIST[@]}"
        do
            PROMPT=t//solve${ROUTE}${NUM}
            TASK_PROMPT=${TASK}${PROMPT}
            yeval \
                --model $MODEL \
                --sample_args "temperature=0.6,top_p=0.9" \
                --task $TASK_PROMPT \
                --include_path routing/ \
                --api_base "http://localhost:${PORT}/v1" \
                --run_name ${MODEL}:${TASK}:${ROUTE}:${NUM} \
                --trust_remote_code \
                --output_path ${SAVE_PATH}/solve_set_train/ $OTHER_ARGS \
                $SYSTEM_ARGS \
        done
    done
done
pkill vllm
