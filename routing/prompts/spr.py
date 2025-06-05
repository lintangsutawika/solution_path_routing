import re

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

from routing.utils import is_runnable_code, extract_answer

from routing.tasks.solve import (
    SolvePL00, SolveNL00,
    SolvePL01, SolveNL01,
    SolvePL02, SolveNL02,
    ) 

# from routing.tasks.route import (
#     Route00,
#     Route01,
#     Route02,
#     Route03,
#     Route04
#     )

PL_list = [
    "Let's write a program to solve this.",
    "I can solve this with python",
    "Let's write a program to solve this."
    ]

NL_list = [
    "The problem is easier to solve by thinking step by step.",
    "I would have more success with chain-of-thought reasoning.",
    "I should solve this by thinking step by step."
]

def routing_function(state, task_list):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"][0].split("\n")[0]
    try:
        if solve_with in PL_list:
            return task_list[0], True
        else:
            return task_list[1], True
    except:
        pass
    if solve_with == "programming language":
        return task_list[0], True
    elif solve_with == "natural language":
        return task_list[1], True
    return None, True


@register_task("solvePL00_NL00")
class SolvePL00NL00(YevalTask):
    subtask_list=[
        SolvePL00,
        SolveNL00
    ]
    subtask_fn=lambda x, y: routing_function(x, y)

@register_task("solvePL00_NL01")
class SolvePL00NL01(SolvePL00NL00):
    subtask_list=[SolvePL00, SolveNL01]

@register_task("solvePL00_NL02")
class SolvePL00NL02(SolvePL00NL00):
    subtask_list=[SolvePL00, SolveNL02]

@register_task("solvePL01_NL00")
class SolvePL01NL00(SolvePL00NL00):
    subtask_list=[SolvePL01, SolveNL00]

@register_task("solvePL01_NL01")
class SolvePL01NL01(SolvePL00NL00):
    subtask_list=[SolvePL01, SolveNL01]

@register_task("solvePL01_NL02")
class SolvePL01NL02(SolvePL00NL00):
    subtask_list=[SolvePL01, SolveNL02]

@register_task("solvePL02_NL00")
class SolvePL02NL00(SolvePL00NL00):
    subtask_list=[SolvePL02, SolveNL00]

@register_task("solvePL02_NL01")
class SolvePL02NL01(SolvePL00NL00):
    subtask_list=[SolvePL02, SolveNL01]

@register_task("solvePL02_NL02")
class SolvePL02NL02(SolvePL00NL00):
    subtask_list=[SolvePL02, SolveNL02]
