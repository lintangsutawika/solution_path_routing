#!/bin/bash
#SBATCH --job-name=solve
#SBATCH --output=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:8
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
    # gsm8k
    # gsm_plus
    # gsm_hard
    math_algebra
    math_counting_and_probability
    math_geometry
    math_intermediate_algebra
    math_number_theory
    math_prealgebra
    math_precalculus
    # mmlu_redux_abstract_algebra
    # mmlu_redux_anatomy
    # mmlu_redux_astronomy
    # mmlu_redux_business_ethics
    # mmlu_redux_clinical_knowledge
    # mmlu_redux_college_biology
    # mmlu_redux_college_chemistry
    # mmlu_redux_college_computer_science
    # mmlu_redux_college_mathematics
    # mmlu_redux_college_medicine
    # mmlu_redux_college_physics
    # mmlu_redux_computer_security
    # mmlu_redux_conceptual_physics
    # mmlu_redux_econometrics
    # mmlu_redux_electrical_engineering
    # mmlu_redux_elementary_mathematics
    # mmlu_redux_formal_logic
    # mmlu_redux_global_facts
    # mmlu_redux_high_school_biology
    # mmlu_redux_high_school_chemistry
    # mmlu_redux_high_school_computer_science
    # mmlu_redux_high_school_european_history
    # mmlu_redux_high_school_geography
    # mmlu_redux_high_school_government_and_politics
    # mmlu_redux_high_school_macroeconomics
    # mmlu_redux_high_school_mathematics
    # mmlu_redux_high_school_microeconomics
    # mmlu_redux_high_school_physics
    # mmlu_redux_high_school_psychology
    # mmlu_redux_high_school_statistics
    # mmlu_redux_high_school_us_history
    # mmlu_redux_high_school_world_history
    # mmlu_redux_human_aging
    # mmlu_redux_human_sexuality
    # mmlu_redux_international_law
    # mmlu_redux_jurisprudence
    # mmlu_redux_logical_fallacies
    # mmlu_redux_machine_learning
    # mmlu_redux_management
    # mmlu_redux_marketing
    # mmlu_redux_medical_genetics
    # mmlu_redux_miscellaneous
    # mmlu_redux_moral_disputes
    # mmlu_redux_moral_scenarios
    # mmlu_redux_nutrition
    # mmlu_redux_philosophy
    # mmlu_redux_prehistory
    # mmlu_redux_professional_accounting
    # mmlu_redux_professional_law
    # mmlu_redux_professional_medicine
    # mmlu_redux_professional_psychology
    # mmlu_redux_public_relations
    # mmlu_redux_security_studies
    # mmlu_redux_sociology
    # mmlu_redux_us_foreign_policy
    # mmlu_redux_virology
    # mmlu_redux_world_religions
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
TASK_LIST=("${6:-${TASK_LIST[@]}}")
ROUTE_LIST=("${7:-${ROUTE_LIST[@]}}")
NUM_LIST=("${8:-${NUM_LIST[@]}}")

MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --enable_prefix_caching \
    --distributed-executor-backend mp > o.txt &

for ROUTE in "${ROUTE_LIST[@]}"
do
    for TASK in "${TASK_LIST[@]}"
    do
        for NUM in "${NUM_LIST[@]}"
        do
            PROMPT=t//solve${ROUTE}${NUM}
            TASK_PROMPT=train:${TASK}${PROMPT}
            for I in {0..7}
            do
                yeval \
                    --model $MODEL \
                    --sample_args "temperature=0.6,top_p=0.9" \
                    --task $TASK_PROMPT \
                    --include_path routing/ \
                    --api_base "http://localhost:${PORT}/v1" \
                    --run_name ${MODEL}:${TASK}:${ROUTE}:${NUM}:${I} \
                    --trust_remote_code \
                    --output_path ${SAVE_PATH}/solve_set_train/ $OTHER_ARGS \
                    $SYSTEM_ARGS
            done
        done
    done
done
pkill vllm
