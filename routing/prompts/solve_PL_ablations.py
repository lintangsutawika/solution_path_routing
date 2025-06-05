import re

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

from routing.utils import is_runnable_code, extract_answer

## A
@register_task("solvePL00A")
class SolvePL00A(YevalTask):
    user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
    postprocessor=lambda x: is_runnable_code(x)
    logging=log_token_usage

@register_task("solvePL01A")
class SolvePL01A(SolvePL00A):
    user_message=lambda x: f"{x}\n\nSolve this by ONLY writing a function called `solution()` that returns a single value to solve the question."

@register_task("solvePL02A")
class SolvePL02A(SolvePL00A):
    user_message=lambda x: f"{x}\n\nWrite ONLY a program. The function must be named solution() without any input arguments and must return a single value."

## B
@register_task("solvePL00B")
class SolvePL00B(YevalTask):
    # user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
    postprocessor=lambda x: is_runnable_code(x)
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's write a Python program to solve it.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }
    logging=log_token_usage

@register_task("solvePL01B")
class SolvePL01B(SolvePL00B):
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's write a code to solve this.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }

@register_task("solvePL02B")
class SolvePL02B(SolvePL00B):
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"Let's solve this with Python.\n```python\ndef solution\(\):\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }

## C
@register_task("solvePL00C")
class SolvePL00C(YevalTask):
    user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
    postprocessor=lambda x: is_runnable_code(x)
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"```python\n(.*?)```",
            "include_stop_str_in_output": True,
            },
    }
    logging=log_token_usage


@register_task("solvePL01C")
class SolvePL01ALT(SolvePL00C):
    user_message=lambda x: f"{x}\n\nSolve this by ONLY writing a function called `solution()` that returns a single value to solve the question."

@register_task("solvePL02C")
class SolvePL02ALT(SolvePL00C):
    user_message=lambda x: f"{x}\n\nWrite ONLY a program. The function must be named solution() without any input arguments and must return a single value."

# ## D
# @register_task("solvePL00C")
# class SolvePL00C(YevalTask):
#     user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
#     postprocessor=lambda x: is_runnable_code(x)
#     sampling_args={
#         "stop": ["```\n"],
#         "extra_body": {
#             "guided_regex": r"```python\n(.*?)```",
#             "include_stop_str_in_output": True,
#             },
#     }
#     logging=log_token_usage

# @register_task("solvePL01C")
# class SolvePL01ALT(SolvePL00C):
#     user_message=lambda x: f"{x}\n\nSolve this by ONLY writing a function called `solution()` that returns a single value to solve the question."

# @register_task("solvePL02C")
# class SolvePL02ALT(SolvePL00C):
#     user_message=lambda x: f"{x}\n\nWrite ONLY a program. The function must be named solution() without any input arguments and must return a single value."


@register_task("solvePL00X")
class SolvePL00X(YevalTask):
    user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."
    postprocessor=lambda x: is_runnable_code(x)
    sampling_args={
        "stop": ["```\n"],
        "extra_body": {
            "guided_regex": r"(.*?)\n```python\ndef solution\(\):\n(.*?)```\n",
            "include_stop_str_in_output": True,
            },
    }
    logging=log_token_usage

@register_task("solvePL01X")
class SolvePL01X(SolvePL00X):
    user_message=lambda x: f"{x}\n\nSolve this by ONLY writing a function called `solution()` that returns a single value to solve the question."

@register_task("solvePL02X")
class SolvePL02X(SolvePL00X):
    user_message=lambda x: f"{x}\n\nWrite ONLY a program. The function must be named solution() without any input arguments and must return a single value."
