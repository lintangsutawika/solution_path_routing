#

## Constructing Ranking Data

```
python routing/ranker.py \
    --data_path evals/solve_set_train/ \
    --output_path data/math/ \
    --task_prefix math_ \
    --model_prefix Qwen-Qwen2.5
```

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