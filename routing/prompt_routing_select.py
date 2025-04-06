from yeval.prompt import YevalPrompt, register_prompt

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
