import re
# from pytorchltr.evaluation import ndcg
from yeval.task import register_task, YevalTask

@register_task("classify")
class RouteClassify(YevalTask):
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's solve this with programming language.",
                "Let's solve this with natural language.",
                ]
        }
    }
