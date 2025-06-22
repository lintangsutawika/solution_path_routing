import os
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
class TabMWPTask(YevalTask):
    data_path="json"
    data_kwargs={
        "data_files": {
            "train": os.path.join(base_url, "problems_train.json"),
            "test": os.path.join(base_url, "problems_test.json"),
            "dev": os.path.join(base_url, "problems_dev.json"),
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
