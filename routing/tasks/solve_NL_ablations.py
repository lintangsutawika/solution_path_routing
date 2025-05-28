import re

from functools import partial
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

from yeval.response import get_boxed_answer
from routing.utils import is_runnable_code, extract_answer

## A
@register_task("solveNL00A")
class SolveNL00A(YevalTask):
    user_message=lambda x: f"{x}\n\nLet's think step by step and write your final answer after 'The answer is'."
    postprocessor=lambda x: extract_answer(x)
    logging=log_token_usage

@register_task("solveNL01A")
class SolveNL01A(SolveNL00A):
    user_message=lambda x: f"{x}\n\nGive step by step reasoning about how to solve the question. Then output the answer after 'The answer is'."

@register_task("solveNL02A")
class SolveNL02A(SolveNL00A):
    user_message=lambda x: f"{x}\n\nFirst give step by step reasoning then write the final answer after 'The answer is'."

## B
@register_task("solveNL00B")
class SolveNL00B(YevalTask):
    postprocessor=get_boxed_answer
    sampling_args={
            "stop": ["}\n"],
            "extra_body": {
                "guided_regex": r"Let's think step by step and write the answer in \\boxed\{\}.(.*?)\\boxed\{(.*)\}\n",
                "include_stop_str_in_output": True,
                },
        }
    logging=log_token_usage

@register_task("solveNL01B")
class SolveNL01B(SolveNL00B):
    sampling_args={
            "stop": ["}\n"],
            "extra_body": {
                "guided_regex": r"Let's reason step by step and write the answer in \\boxed\{\}.(.*?)\\boxed{(.*)}\n",
                "include_stop_str_in_output": True,
                },
        }

@register_task("solveNL02B")
class SolveNL02B(SolveNL00B):
    sampling_args={
            "stop": ["}\n"],
            "extra_body": {
                "guided_regex": r"Let's solve problem step by step and write the answer in \\boxed\{\}.(.*?)\\boxed{(.*)}\n",
                "include_stop_str_in_output": True,
                },
        }

## C
@register_task("solveNL00C")
class SolveNL00C(YevalTask):
    user_message=lambda x: f"{x}"+"\nReason step by step and put your final answer within \\boxed{}."
    postprocessor=get_boxed_answer
    logging=log_token_usage

@register_task("solveNL01C")
class SolveNL01C(SolveNL00C):
    user_message=lambda x: f"{x}"+"\nThink about it step by step and give your answer at the end in \\boxed{}."

@register_task("solveNL02C")
class SolveNL02C(SolveNL00C):
    user_message=lambda x: f"{x}"+"\nFirst give step by step reasoning, then write the answer within \\boxed{}."
