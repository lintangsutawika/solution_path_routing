from yeval.prompt import YevalPrompt, register_prompt

@register_prompt("solve_NL_00")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nLet's think step by step and write the final answer at the end."

@register_prompt("solve_PL_00")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nlet's write a program `solution()` to solve this problem."

@register_prompt("solve_NL_01")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"Give step-by-step reasoning about how to solve the question. Then output the answer."

@register_prompt("solve_PL_01")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"Write a program called `solution()` that returns a single value to solve the question."

@register_prompt("solve_NL_02")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"""\
You will be given a question for you to answer. \
First give step-by-step reasoning about how to solve the question. Then output the answer.
{x}"""

@register_prompt("solve_PL_02")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"""\
You will be given a question for you to answer. \
Firts write a program to solve the problem. \
The function must be named solution() without any input arguments and must return a single value.\
{x}"""
