import re
from yeval.task import register_task, YevalTask
from routing.utils import log_top_n_tokens

@register_task("route_optim00")
class Route00(YevalTask):
    user_message=lambda x: f"{x}"+"Route:"
    logging=log_top_n_tokens
    sampling_args={
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 100,
        "extra_body": {
            "top_k": 100,
            }
        }


