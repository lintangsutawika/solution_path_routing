import re
import sys
import subprocess

import numpy as np

def match_routing(prediction, ground_truth):
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()
    if re.sub(r'[^\w\s]', '', prediction) == re.sub(r'[^\w\s]', '', ground_truth):
        return 1
    elif ground_truth in prediction:
        return 1
    return 0

def process_generation_to_code(gens: str, answer_expr: str):
    if '```python' in gens:
        gens = gens.split('```python')[1].split('```')[0]
    elif '```' in gens:
        gens = gens.split('```')[1].split('```')[0]
    elif answer_expr in gens:
        gens = "def "+answer_expr+f"{answer_expr}".join(gens.split(answer_expr)[1:])
    else:
        return False
        
    return gens.split('\n')

def is_runnable_code(text_string, answer_expr='solution()', time_out=10):
    # Check if the _output is a program
    code = process_generation_to_code(text_string, answer_expr)
    if code:
        def _generate_code(code, answer_expr):
            return "\n".join(code)+f"\nans = 'ans='+str({answer_expr})\nprint(ans)"
        # Generate code snippet that will be executed in a different process
        code_snippet = _generate_code(code, answer_expr)
        try:
            subprocess_result = subprocess.run([sys.executable, "-c", code_snippet], timeout=time_out, text=True, capture_output=True)
            exec_result = subprocess_result.stdout.split("ans=")[-1].strip()
            return exec_result
        except Exception as e:
            return False
    else:
        return False

def extract_bold_text(text):
    return re.findall(r'\*\*(.*?)\*\*', text)

def extract_answer(answer: str):
    try:
        extracted_answer = answer.split('####')[-1].strip()
        answer = answer.strip()
        if extracted_answer == answer:
            # match = re.search(r"answer is(\w)", answer)
            # match = re.search(r"(?i)(?<=answer is ).*", answer)
            # match = re.search(r"(?i)(?<=answer is[:\s]).*", answer)
            match = re.search(r'answer is:?\s*(.*)', answer, re.IGNORECASE)
            
            if match:
                # return match.group(0)
                answer_string = match.group(1).strip()
                extract_string = extract_bold_text(answer_string)
                if len(extract_string) > 0:
                    return extract_string[0].strip()
                else:
                    return answer_string.strip()
            else:
                return answer
        return extracted_answer
    except Exception as e:
        return answer

def preprocess_routing(x, state, pl_system_message, nl_system_message):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"][0].split("\n")[0]
    if solve_with == "programming language":
        state["user_message"] = pl_system_message
    elif solve_with == "natural language":
        state["user_message"] = nl_system_message
    return x, state

def postprocess_routing(x, state):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"][0]
    if solve_with == "programming language":
        code = is_runnable_code(x)
        if code:
            return code, state
        else:
            return x, state
    elif solve_with == "natural language":
        try:
            x = extract_answer(x)
        except:
            pass
    return x, state

def log_top_n_tokens(state):
    for choice in state['choices']:
        top_n_tokens = [token["token"] for token in choice['logprobs']['content'][0]['top_logprobs']]
        return {"top_n_tokens": top_n_tokens}
