import re
from yeval.task import register_task, YevalTask

from yeval.log import log_logprob

@register_task("keyPL00")
class Route00(YevalTask):
    # user_message=lambda x: f"{x}\n\nWhich method is the best way to solve this problem?"
    # sampling_args={"stop": ["\n\n", "\n"], "extra_body":{"guided_choice": ["programming language", "natural language"]}}
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's solve this with programming language.",
                "Let's solve this with natural language.",
                ]
        }
    }

@register_task("route01")
class Route01(Route00):
    # user_message=lambda x: f"{x}\n\nI should solve this by using "
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "I should solve this with Python code.",
                "I should solve this by thinking step by step.",
                ]
        }
    }

@register_task("route02")
class Route02(Route00):
    # user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem (Pick one option)? \"natural language\" or \"programming language\"?"
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "I can solve this with python",
                "I can solve this with reasoning",
                ]
        }
    }

@register_task("route03")
class Route03(Route00):
    # user_message=lambda x: f"{x}\n\nBased on the given task, how should we solve the problem?"
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's write a program to solve this.",
                "Let's write a step by step solution.",
                ]
        }
    }

@register_task("route04")
class Route04(Route00):
    # user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: "
    sampling_args={
        "logprobs": True,
        "extra_body":{
            "guided_choice": [
                "This looks like a computational problem.",
                "This looks like a symbolic reasoning problem."
                ]
        }
    }
    
@register_task("route05")
class Route05(Route00):
    # user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: "
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "The problem is easier to solve with programming.",
                "The problem is harder to solve with programming.",
                ]
        }
    }

@register_task("route06")
class Route06(Route00):
    # user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: "
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "The problem is harder to solve by thinking step by step.",
                "The problem is easier to solve by thinking step by step.",
                ]
        }
    }


@register_task("route07")
class Route07(Route00):
    # user_message=lambda x: f"{x}\n\nChoose only one way that would best solve this problem: "
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "I would have more success with programming.",
                "I would have more success with chain-of-thought reasoning.",
                ]
        }
    }


