from yeval.prompt import YevalPrompt, register_prompt

# @register_prompt("direct_NL_00")
# class DirectNLFirst(YevalPrompt):
#     system_message="""\
# Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
# 1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
# 2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.\
# """

# @register_prompt("direct_PL_00")
# class DirectPLFirst(YevalPrompt):
#     system_message="""\
# Choose only one way to solve the problem: by writing a program OR thinking step-by-step as a way to solve a given task. Do NOT use both:
# 1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
# 2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.\
# """

#     system_message="""\
# Based on a given task, choose only one way that can be used to solve the problem: by programming language OR natural language to solve a given task. Do NOT use both. Answer with either "programming language" or "natural language".
# """

@register_prompt("select_NL_00")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? \"natural language\" or \"programming language\"? Choose only one way."

@register_prompt("select_PL_00")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? \"programming language\" or \"natural language\"? Choose only one way."

@register_prompt("select_NL_01")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"Which method is the best way to solve this problem? You must choose only 1 way. \"natural language\" or \"programming language\"? {x}\nBest way to solve:"

@register_prompt("select_PL_01")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"Which method is the best way to solve this problem? You must choose only 1 way. \"programming language\" or \"natural language\"? {x}\nBest way to solve:"

@register_prompt("select_NL_02")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"natural language\" or \"programming language\"?"

@register_prompt("select_PL_02")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"programming language\" or \"natural language\"?"

@register_prompt("select_NL_03")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem? Remember to only pick one option. \"natural language\" or \"programming language\"?"

@register_prompt("select_PL_03")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem? Remember to only pick one option. \"programming language\" or \"natural language\"?"

@register_prompt("select_NL_04")
class SelectNL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: \"natural language\" or \"programming language\". Answer:"

@register_prompt("select_PL_04")
class SelectPL00(YevalPrompt):
    user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: \"programming language\" or \"natural language\". Answer:"

# @register_prompt("solve_NL_00")
# class SelectNL00(YevalPrompt):
#     system_message="""\
# Think it step by step, and give your answer at the end.
# First give step-by-step reasoning about how to solve the question. Then output the answer.
# """
#     postprocessor="cot"

# @register_prompt("solve_PL_00")
# class SelectPL00(YevalPrompt):
#     system_message="""\
# Based on a given task, choose only one way that can be used to solve the problem: by programming language OR natural language to solve a given task. Do NOT use both. Answer with either "programming language" or "natural language".
# """
#     postprocessor="cot"

# @register_prompt("solve_with_NL-00")
# class SolveWithNL00(YevalPrompt):
#     system_message="""\
# You will be given a question at the end, after the examples, for you to answer. \
# Think it step by step, and give your answer at the end.
# """
#     postprocessor="cot"

# @register_prompt("solve_with_NL-01")
# class SolveWithNL01(YevalPrompt):
#     system_message="""\
# You will be given a question at the end, after the examples, for you to answer. \
# First give step-by-step reasoning about how to solve the question. Then output the answer.
# """
#     postprocessor="cot"
