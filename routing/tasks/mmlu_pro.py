from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.utils import (
        match_routing,
        preprocess_routing,
        postprocess_routing
        )

from functools import partial

letter_choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def input_fn(x):
    text_choice = x["options"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either {",".join(letter_choice)}.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def output_fn(x):
    text_choice = x["options"]
    letter = x['answer']
    label = letter_choice.index(letter)
    text = text_choice[label]
    return [letter, text, f"{letter}. {text}"]

def eval_fn(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        if prediction in letter_choice:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

class MMLUProTask(YevalTask):
    data_path="TIGER-Lab/MMLU-Pro"
    input_text=input_fn
    output_text=output_fn
    test_split="test"
    few_shot_split="validation"
    evaluation={"accuracy": eval_fn}

@register_task("mmlu_pro_routing")
class MMLUProBaseRoutingTask(MMLUProTask):
    sampling_args={"stop": ["\n\n", "\n"]}
    evaluation={"score": lambda x,y: -1}
    logging=log_token_usage

@register_task("mmlu_pro_solve")
class MMLUProRoutingStage(MMLUProTask):
    preprocessor=preprocess_routing
    postprocessor=postprocess_routing
    logging=log_token_usage


if __name__ == "__main__":
    pass
