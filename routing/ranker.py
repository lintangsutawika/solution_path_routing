import os
import fsspec
import argparse
import jsonlines

import numpy as np
import pandas as pd

from functools import partial
from tqdm import tqdm

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

def rank_model_solution_pair(row, coeff=1.0, score_fn=None):

    row = row.unstack(level=0).reset_index()
    row["size"] = row["model"].apply(lambda x: int(x.split("-")[2][:-1]))
    row["compute"] = 2 * row["size"] * row["output_tokens"]

    # row["score"] = row["accuracy"] / (lmbda * np.log10(row["compute"]))
    if score_fn is not None:
        row["score"] = score_fn(row)
    else:
        # row["score"] = row["accuracy"] / (lmbda * np.log10(row["compute"]))
        # row["score"] = np.log10(1+row["accuracy"]/(row["compute"]**lmbda))
        row["score"] = coeff*row['accuracy'] - np.log10(row['compute'])

    row.sort_values(by=['score'], ascending=[False], inplace=True)
    # row.sort_values(by=['accuracy', 'compute'], ascending=[False, True], inplace=True)

    row["pair"] = row["model"] + "-" + row["solve"]

    # return [label_class.index(route_label) for route_label in row["pair"].tolist()]
    return (
        [label_class.index(route_label) for route_label in row["pair"].tolist()],
        row["accuracy"].tolist(),
        row["compute"].tolist(),
        )

    # return row["pair"].tolist()

def rank_solutions(solution_path, model_prefix, task_prefix, use_gcs=False, task_name=False, fs_kwargs={}, save_path=None):

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

        if solution[-3] == ":":
            continue
        model_name, task_name, solve, prompt, num = solution.split(":")

        with fs.open(os.path.join(solution_path, solution, "output.jsonl"), 'r') as f:
            file_df = pd.read_json(f, lines=True)
            df = pd.DataFrame()
            df = file_df[["idx", "accuracy"]].copy()
            df["output_tokens"] = file_df.apply(lambda x: pd.Series(x['step'][0]['log']['output_tokens']), axis=1)
            # task_name_prompt = f"Task: {task_name}" else ""
            # df["text"] = file_df.apply(lambda x: pd.Series(task_name_prompt+x['step'][0]['full_input'][0]['content']), axis=1)
            df["text"] = file_df.apply(lambda x: pd.Series(x['step'][0]['full_input'][0]['content']), axis=1)
            df["task"] = task_name
            df["model"] = model_name
            df["solve"] = solve
            df["prompt"] = prompt
            df["num"] = num
            df["size"] = int([m[:-1] for m in model_name.split("-") if m[0].isdigit()][0])

        solution_df = pd.concat([solution_df, df], ignore_index=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        solution_df.to_json(save_path, orient="records", lines=True)

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
    args.add_argument("--task_name", type=str, help="Flag to indicate whether to use Google Cloud Storage.")
    args.add_argument("--coeff", type=float, default=0.1, help="Lambda value for ranking score calculation.")
    args.add_argument("--save_df_path", type=str, default=None, help="Path to save the DataFrame.")
    args = args.parse_args()

    if (args.save_df_path is not None) and (os.path.exists(args.save_df_path)):
        print(f"DataFrame already exists at {args.save_df_path}. Loading from there.")
        df = pd.read_json(args.save_df_path, lines=True)
    else:
        df = rank_solutions(
            # "gs://cmu-gpucloud-lsutawik/spr/solve_set_train",
            args.data_path,
            model_prefix=args.model_prefix,
            task_prefix=args.task_prefix,
            use_gcs=args.use_gcs,
            task_name=args.task_name,
            save_path=args.save_df_path,
            # fs_kwargs = {k: v for kv in args.fs_kwargs.split(",") if kv for k, v in [kv.split("=")]}
            # fs_kwargs={
            #     "project":'cmu-gpu-cloud-flame',
            #     "token":'/home/lsutawik/.config/gcloud/application_default_credentials.json'
            #     }
            )
    print(df.head())
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    task_df = df[df.task.str.contains(args.task_prefix)]
    for task_name in task_df.task.unique():
        _task_df = task_df[task_df["task"] == task_name]
        _rank_df = _task_df.pivot_table(
            index=["idx"],
            columns=["model", "solve"],
            values=["accuracy", "output_tokens"],
            aggfunc="mean",
        )

        _task_df.reset_index(inplace=True)

        rank_fn = partial(rank_model_solution_pair, coeff=args.coeff)
        ranked_lists = _rank_df.apply(rank_fn, axis=1)
        _rank_df["label"], _rank_df["rank_accuracy"], _rank_df["rank_compute"] = zip(*ranked_lists)
        _rank_df.reset_index(inplace=True)

        _rank_df.columns = ['_'.join(filter(None, col)).strip() for col in _rank_df.columns.values]
        _rank_df = _rank_df[['idx', 'label', 'rank_accuracy', 'rank_compute']]
        # _rank_df["route"] = _rank_df["label"].apply(lambda x: label_class[x[0]])
        # _rank_df["model"] = _rank_df["route"].apply(lambda x: "-".join(x.split("-")[:-1]))
        # _rank_df["solve"] = _rank_df["route"].apply(lambda x: x.split("-")[-1])

        _text_df = _task_df[
            (_task_df["model"] == _task_df["model"].unique()[0])
            & (_task_df["solve"] == "PL")
            & (_task_df["prompt"] == 0)
            & (_task_df["num"] == 0)
        ]

        # print(_text_df.head())
        _rank_df = _text_df[['idx', 'text']].merge(_rank_df.reset_index(drop=True)[['label', 'idx', 'rank_accuracy', 'rank_compute']], on='idx')
        _rank_df["task"] = task_name

        _rank_df.sample(frac=1).reset_index(drop=True)
        task_size = len(_rank_df)
        train_size = int(task_size * 0.8)
        valid_size = int(task_size * 0.1)
        test_size = int(task_size * 0.1)
        train_df = pd.concat([train_df, _rank_df.iloc[:train_size]], ignore_index=True)
        valid_df = pd.concat([valid_df, _rank_df.iloc[train_size:train_size + valid_size]], ignore_index=True)
        test_df = pd.concat([test_df, _rank_df.iloc[-test_size-1:]], ignore_index=True)

    print(train_df.head())

    os.makedirs(args.output_path, exist_ok=True)
    train_df.to_json(os.path.join(args.output_path, "train.jsonl"), orient="records", lines=True)
    valid_df.to_json(os.path.join(args.output_path, "valid.jsonl"), orient="records", lines=True)
    test_df.to_json(os.path.join(args.output_path, "test.jsonl"), orient="records", lines=True)