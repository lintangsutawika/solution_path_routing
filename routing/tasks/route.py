import re

from yeval.task import register_task, YevalTask

from routing.utils import (
    match_routing,
    )

@register_task("route00")
class Route00(YevalTask):
    user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem?"
    sampling_args={"stop": ["\n\n", "\n"], "extra_body":{"guided_choice": ["programming language", "natural language"]}}

@register_task("route01")
class Route01(Route00):
    user_message=lambda x: f"{x}\n\nI should solve this by using "

@register_task("route02")
class Route01(Route00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"natural language\" or \"programming language\"?"

@register_task("route03")
class Route01(Route00):
    user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem?"

@register_task("route04")
class Route01(Route00):
    user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: "
