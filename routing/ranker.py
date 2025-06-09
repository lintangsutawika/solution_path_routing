import os
import fsspec
import argparse
import jsonlines

import pandas as pd

from tqdm import tqdm

def rank_solutions(solution_path, model_prefix, task_prefix, use_gcs=False, fs_kwargs={}):

    if use_gcs:
        fs = fsspec.filesystem('gcs', **fs_kwargs)
    else:
        fs = fsspec.filesystem('file')

    solution_list = [file_path.split("/")[-1] for file_path in fs.ls(solution_path)]

    solution_df = pd.DataFrame()

    for solution in tqdm(solution_list):
        if task_prefix not in solution:
            continue

        if model_prefix not in solution:
            continue

        model_name, task_name, solve, num = solution.split(":")

        with fs.open(os.path.join(solution_path, solution, "output.jsonl"), 'r') as f:
            file_df = pd.read_json(f, lines=True)
            df = pd.DataFrame()
            df = file_df[["idx", "accuracy"]].copy()
            df["output_tokens"] = file_df.apply(lambda x: pd.Series(x['step'][0]['log']['output_tokens']), axis=1)
            df["text"] = file_df.apply(lambda x: pd.Series(x['step'][0]['full_input'][0]['content']), axis=1)
            df["task"] = task_name
            df["model"] = model_name
            df["solve"] = solve
            df["num"] = num
            df["size"] = int([m[:-1] for m in model_name.split("-") if m[0].isdigit()][0])

        solution_df = pd.concat([solution_df, df], ignore_index=True)

    return solution_df


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Router to Rank")
    args.add_argument("--data_path", type=str, default=None, help="Path to save the output.")
    args.add_argument("--output_path", type=str, default=None, help="Path to save the output.")
    args.add_argument("--model_prefix", type=str, required=True, help="Prefix of the model to filter solutions.")
    args.add_argument("--task_prefix", type=str, required=True, help="Prefix of the task to filter solutions.")
    args.add_argument("--save_model_path", type=str, default=None, help="Path to save the model.")
    args.add_argument("--fs_kwargs", type=str, default="", help="Comma-separated key=value pairs for filesystem kwargs.")
    args.add_argument("--use_gcs", action="store_true", help="Flag to indicate whether to use Google Cloud Storage.")
    args = args.parse_args()

    df = rank_solutions(
        # "gs://cmu-gpucloud-lsutawik/spr/solve_set_train",
        args.data_path,
        model_prefix=args.model_prefix,
        task_prefix=args.task_prefix,
        use_gcs=args.use_gcs,
        # fs_kwargs = {k: v for kv in args.fs_kwargs.split(",") if kv for k, v in [kv.split("=")]}
        # fs_kwargs={
        #     "project":'cmu-gpu-cloud-flame',
        #     "token":'/home/lsutawik/.config/gcloud/application_default_credentials.json'
        #     }
        )
    print(df.head())

    label_class = [
        "Qwen-Qwen2.5-3B-Instruct-PL", # 0
        "Qwen-Qwen2.5-3B-Instruct-NL", # 1
        "Qwen-Qwen2.5-7B-Instruct-PL", # 2
        "Qwen-Qwen2.5-7B-Instruct-NL", # 3
        "Qwen-Qwen2.5-14B-Instruct-PL",# 4
        "Qwen-Qwen2.5-14B-Instruct-NL",# 5
        "Qwen-Qwen2.5-32B-Instruct-PL",# 6
        "Qwen-Qwen2.5-32B-Instruct-NL",# 7
        "Qwen-Qwen2.5-72B-Instruct-PL",# 8
        "Qwen-Qwen2.5-72B-Instruct-NL",# 9
    ]
    
    task_df = df[df.task.str.contains("math_")]
    task_df = task_df.pivot_table(
        index=["idx"],
        columns=["model", "solve"],
        values=["accuracy", "output_tokens"],
        aggfunc="mean",
    )

    #                        model solve  accuracy  output_tokens
    # 3  Qwen-Qwen2.5-32B-Instruct    PL  1.000000     208.666667
    # 5   Qwen-Qwen2.5-3B-Instruct    PL  1.000000     348.333333
    # 2  Qwen-Qwen2.5-32B-Instruct    NL  1.000000     534.333333
    # 6  Qwen-Qwen2.5-72B-Instruct    NL  1.000000     640.666667
    # 8   Qwen-Qwen2.5-7B-Instruct    NL  1.000000     655.333333
    # 0  Qwen-Qwen2.5-14B-Instruct    NL  1.000000     658.333333
    # 7  Qwen-Qwen2.5-72B-Instruct    PL  0.666667     216.000000
    # 9   Qwen-Qwen2.5-7B-Instruct    PL  0.666667     233.666667
    # 4   Qwen-Qwen2.5-3B-Instruct    NL  0.666667     650.333333
    # 1  Qwen-Qwen2.5-14B-Instruct    PL  0.333333     238.333333

    def rank_model_solution_pair(row):

        row = row.unstack(level=0).reset_index()
        row["size"] = row["model"].apply(lambda x: int(x.split("-")[2][:-1]))
        row["compute"] = 2 * row["size"] * row["output_tokens"]
        row.sort_values(by=['accuracy', 'compute'], ascending=[False, True], inplace=True)

        row["pair"] = row["model"] + "-" + row["solve"]

        return [label_class.index(route_label) for route_label in row["pair"].tolist()]
        # return row["pair"].tolist()

    # task_df[["value", "value"]] = task_df.apply(rank_model_solution_pair, axis=1)
    task_df["rank"] = task_df.apply(rank_model_solution_pair, axis=1)
    task_df["idx"] = task_df.index
    # task_df[['idx', 'rank']].iloc[0]
    task_df.columns = ['_'.join(filter(None, col)).strip() for col in task_df.columns.values]


    merged_df = df[['idx', 'text']].merge(task_df.reset_index(drop=True)[['rank', 'idx']], on='idx')
    merged_df.to_json()

    train_df = merged_df.sample(frac=0.9, random_state=42)
    test_df = merged_df.drop(train_df.index)

    os.makedirs(args.output_path, exist_ok=True)
    train_df.to_json(os.path.join(args.output_path, "train.jsonl"), orient="records", lines=True)
    test_df.to_json(os.path.join(args.output_path, "test.jsonl"), orient="records", lines=True)