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
        fs_kwargs = {k: v for kv in args.fs_kwargs.split(",") if kv for k, v in [kv.split("=")]}
        # fs_kwargs={
        #     "project":'cmu-gpu-cloud-flame',
        #     "token":'/home/lsutawik/.config/gcloud/application_default_credentials.json'
        #     }
        )

    for task in df["task"].unique():
        




# df.pivot_table(
#     index=["task", "size"],
#     columns=["solve"],
#     # values=["accuracy", "output_tokens"],
#     aggfunc=sum,
# ).reset_index().to_csv("solution_summary.csv", index=False)
