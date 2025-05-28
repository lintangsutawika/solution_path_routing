import re
from yeval.task import register_task, YevalTask

from yeval.log.usage import log_logprob

PL={
    "00": "Let's solve this with programming language.",
    "01": "I should solve this with Python code.",
    "02": "I can solve this with python",
    "03": "Let's write a program to solve this.",
    "04": "This looks like a computational problem.",
    "05": "The problem is easier to solve with programming.",
    "06": "The problem is harder to solve by thinking step by step.",
    "07": "I would have more success with programming.",
}

NL={
    "00": "Let's solve this with natural language.",
    "01": "I should solve this by thinking step by step.",
    "02": "I can solve this with reasoning",
    "03": "Let's write a step by step solution.",
    "04": "This looks like a symbolic reasoning problem.",
    "05": "The problem is harder to solve with programming.",
    "06": "The problem is easier to solve by thinking step by step.",
    "07": "I would have more success with chain-of-thought reasoning.",
}

# PL 00 - NL 05
@register_task("route00")
class Route00(YevalTask):
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's solve this with programming language.",
                "The problem is harder to solve with programming.",
                ]
        }
    }

# PL 00 - NL 06
@register_task("route01")
class Route01(YevalTask):
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's solve this with programming language.",
                "The problem is easier to solve by thinking step by step.",
                ]
        }
    }

# PL 03 - NL 05
@register_task("route02")
class Route02(YevalTask):
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's write a program to solve this.",
                "The problem is harder to solve with programming.",
                ]
        }
    }

# PL 03 - NL 06
@register_task("route03")
class Route03(YevalTask):
    sampling_args={
        "extra_body":{
            "guided_choice": [
                "Let's write a program to solve this.",
                "The problem is easier to solve by thinking step by step.",
                ]
        }
    }

@register_task("getLogProbPL00NL00")
class LogProbPL00(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["00"]]},
    }

@register_task("getLogProbPL00NL01")
class LogProbPL01(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["01"]]},
    }

@register_task("getLogProbPL00NL02")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["02"]]},
    }

@register_task("getLogProbPL00NL03")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["03"]]},
    }

@register_task("getLogProbPL00NL04")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["04"]]},
    }

@register_task("getLogProbPL00NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["05"]]},
    }

@register_task("getLogProbPL00NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["06"]]},
    }

@register_task("getLogProbPL00NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["00"], NL["07"]]},
    }

@register_task("getLogProbPL01NL00")
class LogProbPL01(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["00"]]},
    }

@register_task("getLogProbPL01NL01")
class LogProbPL01(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["01"]]},
    }

@register_task("getLogProbPL01NL02")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["02"]]},
    }

@register_task("getLogProbPL01NL03")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["03"]]},
    }

@register_task("getLogProbPL01NL04")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["04"]]},
    }

@register_task("getLogProbPL01NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["05"]]},
    }

@register_task("getLogProbPL01NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["06"]]},
    }

@register_task("getLogProbPL01NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["01"], NL["07"]]},
    }

@register_task("getLogProbPL02NL00")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["00"]]},
    }

@register_task("getLogProbPL02NL01")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["01"]]},
    }

@register_task("getLogProbPL02NL02")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["02"]]},
    }

@register_task("getLogProbPL02NL03")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["03"]]},
    }

@register_task("getLogProbPL02NL04")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["04"]]},
    }

@register_task("getLogProbPL02NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["05"]]},
    }

@register_task("getLogProbPL02NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["06"]]},
    }

@register_task("getLogProbPL02NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["02"], NL["07"]]},
    }

@register_task("getLogProbPL03NL00")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["00"]]},
    }

@register_task("getLogProbPL03NL01")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["01"]]},
    }

@register_task("getLogProbPL03NL02")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["02"]]},
    }

@register_task("getLogProbPL03NL03")
class LogProbPL03(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["03"]]},
    }

@register_task("getLogProbPL03NL04")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["04"]]},
    }

@register_task("getLogProbPL03NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["05"]]},
    }

@register_task("getLogProbPL03NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["06"]]},
    }

@register_task("getLogProbPL03NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["03"], NL["07"]]},
    }

@register_task("getLogProbPL04NL00")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["00"]]},
    }

@register_task("getLogProbPL04NL01")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["01"]]},
    }

@register_task("getLogProbPL04NL02")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["02"]]},
    }

@register_task("getLogProbPL04NL03")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["03"]]},
    }

@register_task("getLogProbPL04NL04")
class LogProbPL04(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["04"]]},
    }

@register_task("getLogProbPL04NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["05"]]},
    }

@register_task("getLogProbPL04NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["06"]]},
    }

@register_task("getLogProbPL04NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["04"], NL["07"]]},
    }

@register_task("getLogProbPL05NL00")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["00"]]},
    }

@register_task("getLogProbPL05NL01")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["01"]]},
    }

@register_task("getLogProbPL05NL02")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["02"]]},
    }

@register_task("getLogProbPL05NL03")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["03"]]},
    }

@register_task("getLogProbPL05NL04")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["04"]]},
    }

@register_task("getLogProbPL05NL05")
class LogProbPL05(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["05"]]},
    }

@register_task("getLogProbPL05NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["06"]]},
    }

@register_task("getLogProbPL05NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["05"], NL["07"]]},
    }

@register_task("getLogProbPL06NL00")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["00"]]},
    }

@register_task("getLogProbPL06NL01")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["01"]]},
    }

@register_task("getLogProbPL06NL02")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["02"]]},
    }

@register_task("getLogProbPL06NL03")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["03"]]},
    }

@register_task("getLogProbPL06NL04")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["04"]]},
    }

@register_task("getLogProbPL06NL05")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["05"]]},
    }

@register_task("getLogProbPL06NL06")
class LogProbPL06(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["06"]]},
    }

@register_task("getLogProbPL06NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["06"], NL["07"]]},
    }

@register_task("getLogProbPL07NL00")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["00"]]},
    }

@register_task("getLogProbPL07NL01")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["01"]]},
    }

@register_task("getLogProbPL07NL02")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["02"]]},
    }

@register_task("getLogProbPL07NL03")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["03"]]},
    }

@register_task("getLogProbPL07NL04")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["04"]]},
    }

@register_task("getLogProbPL07NL05")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["05"]]},
    }

@register_task("getLogProbPL07NL06")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["06"]]},
    }

@register_task("getLogProbPL07NL07")
class LogProbPL07(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {"guided_choice": [PL["07"], NL["07"]]},
    }

@register_task("testRoute00")
class LogProbPL02(YevalTask):
    user_message=lambda x: f"{x}"+"\nThink about the best way to solve this problem."
    logging=log_logprob
    sampling_args={
        "temperature": 0,
        "logprobs" : True,
        "extra_body": {
            # "guided_decoding_backend": "lm-format-enforcer",
            "guided_choice": [
            # # GSM Hard
            # """It look a while as  there likely meant""",
            # """From basic concepts understanding about ratio we divide information""",
            # Math Algebra
            # """A way around setting, a = The two lines""",
            # """Let 'm''s denote an arbitrary number""",
            # """Astrig conditions set require functions represented into to""",
            # """Let 'm’ stand the number the""",
            # """F the""",
            # """Taking to"""
            """F首先 ，我们现在的需求之一便是需得画""",
            """Taking such logarithmi-free fractional-retract based exponential"""
            ]
        },
    }