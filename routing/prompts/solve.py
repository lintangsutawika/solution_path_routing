import re

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

from yeval.response import get_boxed_answer

from routing.utils import is_runnable_code, extract_answer

@register_task("solvePL00")
class SolvePL00(YevalTask):
    user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
    postprocessor=lambda x: is_runnable_code(x)
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's write a Python program to solve it.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }
    logging=log_token_usage

@register_task("solvePL01")
class SolvePL01(SolvePL00):
    user_message=lambda x: f"{x}\n\nSolve this by ONLY writing a function called `solution()` that returns a single value to solve the question."
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's write a code to solve this.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }

@register_task("solvePL02")
class SolvePL02(SolvePL00):
    user_message=lambda x: f"{x}\n\nWrite ONLY a program. The function must be named solution() without any input arguments and must return a single value."
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's solve this with Python.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }

@register_task("solveNL00")
class SolveNL00(YevalTask):
    user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."
    postprocessor=get_boxed_answer
    logging=log_token_usage

@register_task("solveNL01")
class SolveNL01(SolveNL00):
    user_message=lambda x: f"{x}"+"\nThink about it step by step and give your answer at the end in \\boxed{}."

@register_task("solveNL02")
class SolveNL02(SolveNL00):
    user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning, then write the answer within \\boxed{}."

