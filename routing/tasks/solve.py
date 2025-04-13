import re

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

from routing.utils import (
    preprocess_routing,
    postprocess_routing
    )

@register_task("solve00")
class Solve00(YevalTask):
    preprocessor=partial(
        preprocess_routing,
        pl_system_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`.",
        nl_system_message=lambda x: f"{x}\n\nLet's think step by step and write your final answer after `Answer:`."
    )
    postprocessor=postprocess_routing
    logging=log_token_usage
    evaluation={"accuracy": math_eval}

@register_task("solve01")
class Solve01(Solve00):
    preprocessor=partial(
        preprocess_routing,
        pl_system_message=lambda x: f"{x}\n\nWrite a program called `solution()` that returns a single value to solve the question.",
        nl_system_message=lambda x: f"{x}\n\nGive step-by-step reasoning about how to solve the question. Then output the answer after `Answer:`."
    )

@register_task("solve02")
class Solve01(Solve00):
    preprocessor=partial(
        preprocess_routing,
        pl_system_message=lambda x: f"{x}\n\nWrite a program. The function must be named solution() without any input arguments and must return a single value.",
        nl_system_message=lambda x: f"{x}\n\nFirst give step-by-step reasoning about how to solve the question. Then output the answer after `Answer:`."
    )
