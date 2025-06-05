from yeval.prompt import YevalPrompt, register_prompt

@register_prompt("direct_NL_00")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"""\
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
{x}
"""

@register_prompt("direct_PL_00")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"""\
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
{x}
"""

@register_prompt("direct_NL_01")
class SelectNL00(YevalPrompt):
    system_message=lambda x: f"""\
{x}
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
"""

@register_prompt("direct_PL_01")
class SelectPL00(YevalPrompt):
    system_message=lambda x: f"""\
{x}
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
"""
