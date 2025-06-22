import os
from yeval.task import register_task, YevalTask

dir_path = os.path.dirname(os.path.realpath(__file__))

def finqa_input(x):

    pre_text = "\n".join(x['pre_text'])
    post_text = "\n".join(x['post_text'])
    table = "\n".join([" | ".join(line) for line in x['table']])
    question = x['question']

    return f"{pre_text}\n\nTable:\n{table}\n\n{post_text}\n{question}\nAnswer"

def finqa_output(x):
    return x['answer']

def match_decimals(prediction, ground_truth):
    reversed_number = str(ground_truth)[::-1]
    decimal_places = reversed_number.find('.')
    decimal_places = decimal_places if decimal_places != -1 else 0
    if decimal_places == 1 and reversed_number[0] == "0":
        decimal_places = 0
    rounded_prediction = round(prediction, decimal_places)
    return rounded_prediction


def finqa_eval(prediction, ground_truth):
    try:

        if ground_truth in ["yes", "no"]:
            if prediction == "True":
                prediction = "yes"
            elif prediction == "False":
                prediction = "no"

            score = 1 if prediction == ground_truth else 0

        else:
            prediction = float(prediction)
            if "%" in ground_truth:
                ground_truth = ground_truth.replace("%", "")
                if prediction < 1.0:
                    prediction = prediction * 100    
                
            ground_truth = float(ground_truth)
            prediction = match_decimals(prediction, ground_truth)
            score = 1 if abs(prediction - ground_truth) < 1e-3 else 0

    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

class FinQATask(YevalTask):
    data_path="ibm-research/finqa"
    input_text=finqa_input
    output_text=finqa_output
    evaluation={"accuracy": finqa_eval}

@register_task("train:finqa")
class FinQATrainTask(FinQATask):
    test_split="train"

@register_task("dev:finqa")
class FinQATrainTask(FinQATask):
    test_split="validation"

@register_task("test:finqa")
class FinQATrainTask(FinQATask):
    test_split="test"
