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
