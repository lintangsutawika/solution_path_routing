from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

from yeval.task.arc import ARCChallengeTask

@register_task("train:arc_challenge")
class TrainARCChallengeTask(ARCChallengeTask):
    test_split="train"
