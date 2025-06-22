#

## Constructing Ranking Data

```
python routing/ranker.py \
    --data_path evals/solve_set_train/ \
    --output_path data/math/ \
    --task_prefix math_ \
    --model_prefix Qwen-Qwen2.5
```

for LAMBDA in 0.1 0.2 0.4 0.8 1.0; do python routing/ranker.py --data_path evals/solve_set_train/ --output_path data/math:${LAMBDA}/ --task_prefix math --model_prefix Qwen-Qwen2.5 --lmbda ${LAMBDA}; done

## Training Routing Model

```
python routing/rank_train.py \
    --data_dir data/math/ \
    --model_name Qwen/Qwen2.5-0.5-Instruct \
    --num_labels 10 \
    --output_dir /path/to/output/ \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4
```