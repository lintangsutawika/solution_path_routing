import os
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

from yeval.task.gsm8k import GSM8KTask

@register_task("train:gsm_plus")
class TrainGSMPLUSTask(YevalTask):
    data_path="qintongli/GSM-Plus"
    input_text=lambda x: "Question: " + x["question"] + "\nAnswer:"
    output_text=lambda x: x["answer"]
    test_split="testmini"
    evaluation={"accuracy": math_eval}

@register_task("train:gsm_hard")
class TrainGSM8HardTask(YevalTask):
    data_path="reasoning-machines/gsm-hard"
    input_text=lambda x: "Question: " + x["input"] + "\nAnswer:"
    output_text=lambda x: x["target"]
    test_split="train"
    evaluation={"accuracy": math_eval}

@register_task("train:gsm8k")
class TrainGSM8KTask(GSM8KTask):
    test_split="train"

@register_task("test:gsm8k")
class TrainGSM8KTask(GSM8KTask):
    test_split="test"

@register_task("train:svamp")
class TrainSVAMPTask(YevalTask):
    data_path="json"
    data_kwargs={
        "data_files": {
            "train": os.path.join(
                "https://raw.githubusercontent.com/arkilpatel/SVAMP/refs/heads/main/",
                "SVAMP.json"
                ),
            }
        }
    input_text=lambda x: f"{x["Body"]} {x["Question"]}"
    output_text=lambda x: x["Answer"]
    test_split="train"
    evaluation={"accuracy": math_eval}

GSM_IC_URL = "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/refs/heads/main/"

@register_task("train:gsm_ic_2_step")
class TrainGSMIC2StepTask(YevalTask):
    data_path="json"
    data_kwargs={
        "data_files": {
            "train": os.path.join(
                GSM_IC_URL,
                "GSM-IC_2step.json"
                ),
            }
        }
    input_text=lambda x: x["new_question"]
    output_text=lambda x: x["answer"]
    test_split="train"
    evaluation={"accuracy": math_eval}

@register_task("test:gsm_ic_m_step")
class TrainGSMICMStepTask(TrainGSMIC2StepTask):
    data_kwargs={
        "data_files": {
            "train": os.path.join(
                GSM_IC_URL,
                "GSM-IC_mstep.json"
                ),
            }
        }

