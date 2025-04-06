import re
from functools import partial

from yeval.task import register_task, YevalTask
from yeval.task.commonsense_qa import CommonsenseQATask

from yeval.log.usage import log_token_usage
from routing.utils import (
    match_routing,
    preprocess_routing,
    postprocess_routing
    )

@register_task("commonsenseqa_routing")
class CommonsenseQABaseRoutingTask(CommonsenseQATask):
    input_text=lambda x: x['question']
    output_text=lambda x: "natural language"
    sampling_args={"stop": ["\n\n", "\n"]}
    evaluation={"accuracy": match_routing}
    logging=log_token_usage
