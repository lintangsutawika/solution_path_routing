from yeval.task.mmlu_pro import MMLUProTask
from yeval.task import register_task, YevalTask

@register_task("train:mmlu_pro")
class TrainMMLUProTask(MMLUProTask):
    test_split='validation'

@register_task("test:mmlu_pro")
class TrainMMLUProTask(MMLUProTask):
    pass
