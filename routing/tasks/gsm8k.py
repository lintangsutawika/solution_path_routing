import re
from functools import partial

from yeval.task import register_task, YevalTask
from yeval.task.gsm8k import GSM8KTask

from yeval.log.usage import log_token_usage
from routing.utils import (
    match_routing,
    preprocess_routing,
    postprocess_routing
    )

from yeval.metrics import math_eval

# @register_task("gsm8k_direct_nl_first_A")
# @register_task("gsm8k_direct_nl_first_B")
# @register_task("gsm8k_direct_nl_first_C")
# @register_task("gsm8k_direct_nl_first_D")
# @register_task("gsm8k_direct_pl_first_A")
# @register_task("gsm8k_direct_pl_first_B")
# @register_task("gsm8k_direct_pl_first_C")
# @register_task("gsm8k_direct_pl_first_D")

@register_task("gsm8k_routing")
class GSM8KBaseRoutingTask(GSM8KTask):
    input_text=lambda x: x['question']
    output_text=lambda x: "programming language"
    sampling_args={"stop": ["\n\n", "\n"]}
    evaluation={"accuracy": match_routing}

# @register_task("gsm8k_solve")
# class GSM8KRoutingStage(GSM8KTask):
#     preprocessor=preprocess_routing
#     postprocessor=postprocess_routing


# class GSM8KRoutingATask(YevalTask):
#    subtask_list=[
#        GSM8KRoutingNLFirstTask,
#        GSM8KRoutingStage
#    ]

# @register_task("gsm8k_routing_pl_first")
# class GSM8KRoutingBTask(YevalTask):
#    subtask_list=[
#        GSM8KRoutingPLFirstTask,
#        GSM8KRoutingStage
#    ]
