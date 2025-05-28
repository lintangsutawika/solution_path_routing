from yeval.task import register_task, YevalTask

@register_task("selectNL00")
class SelectNL00(YevalTask):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? \"natural language\" or \"programming language\"? Choose only one way."
    sampling_args={
        "temperature": 0,
        "extra_body":{
            "guided_choice": [
                "programming language",
                "natural language",
            ]
        }
    }

@register_task("selectPL00")
class SelectPL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? \"programming language\" or \"natural language\"? Choose only one way."

@register_task("selectNL01")
class SelectNL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? You must choose only 1 way. \"natural language\" or \"programming language\"? Best way to solve:"

@register_task("selectPL01")
class SelectPL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem? You must choose only 1 way. \"programming language\" or \"natural language\"? Best way to solve:"

@register_task("selectNL02")
class SelectNL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"natural language\" or \"programming language\"?"

@register_task("selectPL02")
class SelectPL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"programming language\" or \"natural language\"?"

@register_task("selectNL03")
class SelectNL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem? Remember to only pick one option. \"natural language\" or \"programming language\"?"

@register_task("selectPL03")
class SelectPL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem? Remember to only pick one option. \"programming language\" or \"natural language\"?"

@register_task("selectNL04")
class SelectNL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: \"natural language\" or \"programming language\". Answer:"

@register_task("selectPL04")
class SelectPL00(SelectNL00):
    user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: \"programming language\" or \"natural language\". Answer:"
