from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

from yeval.task.tabmwp import TabMWPTask

@register_task("train:tabmwp")
class TrainTabMWPTask(TabMWPTask):
    test_split="dev"