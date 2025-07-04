import os
import pandas as pd
from yeval.task import register_task, YevalTask

dir_path = os.path.dirname(os.path.realpath(__file__))

def tabmwp_input(x):
    return "Table:\n"+x['table']+"\nQuestion:\n"+x['question'].strip()+"\nAnswer:"

def tabmwp_output(x):
    return x["answer"]

def tabmwp_eval(prediction, ground_truth):
    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

base_url = "https://raw.githubusercontent.com/lupantech/PromptPG/refs/heads/main/data/tabmwp/"
for split in ["train", "test", "dev"]:
    file_path = os.path.join(dir_path, "tabmwp/", f"problems_{split}.parquet")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        # pd.read_json(f"{base_url}problems_{split}.json").transpose().to_json(file_path, lines=True, orient="records")
        pd.read_json(f"{base_url}problems_{split}.json").transpose()[
            ["question", "answer", "choices", "ques_type", "table", "grade"]
             ].to_parquet(file_path, index=False)

class TabMWPTask(YevalTask):
    data_path="parquet"
    data_kwargs={
        "data_files": {
            # "train": os.path.join(base_url, "problems_train.json"),
            # "test": os.path.join(base_url, "problems_test.json"),
            # "dev": os.path.join(base_url, "problems_dev.json"),
            "train": os.path.join(dir_path, "tabmwp/", "problems_train.parquet"),
            "test": os.path.join(dir_path, "tabmwp/", "problems_test.parquet"),
            "dev": os.path.join(dir_path, "tabmwp/", "problems_dev.parquet"),
            }
        }
    input_text=tabmwp_input
    output_text=tabmwp_output
    evaluation={"accuracy": tabmwp_eval}

@register_task("train:tabmwp")
class TabMWPTrainTask(TabMWPTask):
    test_split="train"

@register_task("dev:tabmwp")
class TabMWPTrainTask(TabMWPTask):
    test_split="dev"

@register_task("test:tabmwp")
class TabMWPTrainTask(TabMWPTask):
    test_split="test"
