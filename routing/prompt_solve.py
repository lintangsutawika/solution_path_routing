from yeval.prompt import YevalPrompt, register_prompt
from yeval.response.code_responses import is_runnable_code

@register_prompt("solve_NL_00")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nLet's think step by step and write your final answer after `Answer:`."

@register_prompt("solve_PL_00")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nlet's solve this by DIRECTLY and ONLY writing a program `solution()`."

@register_prompt("solve_NL_01")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nGive step-by-step reasoning about how to solve the question. Then output the answer after `Answer:`."

@register_prompt("solve_PL_01")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nWrite a program called `solution()` that returns a single value to solve the question."

@register_prompt("solve_NL_02")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nFirst give step-by-step reasoning about how to solve the question. Then output the answer after `Answer:`."

@register_prompt("solve_PL_02")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nWrite a program. The function must be named solution() without any input arguments and must return a single value."