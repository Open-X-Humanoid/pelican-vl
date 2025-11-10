import os
import time

import numpy as np
from openai import OpenAI
from lmms_eval.cot_utils import remove_think

base_url = os.environ.get('JUDGE_OPENAI_API_BASE', None)
judge_model = OpenAI(api_key=os.environ.get('JUDGE_OPENAI_API_KEY', ""), base_url=base_url)

def build_prompt(question, options, prediction):
    """
    Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

    If the GPT-3.5 model is unable to find a match, it will output (Z).
    Also, if the original prediction does not clearly lean towards any of the options, it will output (Z).

    Parameters:
    - question: String, the question.
    - options: String, the options. E.g. ['(A)', '(B)']
    - prediction: String, the answer. E.g. '(B)'
    """
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
        "Your should output one of the choices, (A),(B),(C),(D),(E) (if they are valid options), or (Z)\n"
        "Example 1: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
        "Example 2: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        "Example 3: \n"
        "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
        "Your Round: \n"
        "Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)

def match_multiple_choice(question, options, prediction):
    prompt = build_prompt(question, options, prediction)
    retry_limit = 10

    for retry in range(retry_limit):
        try:
            response = judge_model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
            for choice in choices:
                if choice in content:
                    return choice
            return content
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    return '(Z) Failed to get multiple choice'

def analyze_answer(doc, gpt_answer, all_choices):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - d : data, the data containing the question and choices.
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    try:
        intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
        intersect_last = list(set(all_choices).intersection(set(gpt_answer.split('\n\n')[-1].split())))
        if gpt_answer in ["A", "B", "C", "D", "E"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            prediction = gpt_answer
        elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
            choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
            options = '\n'.join([f'{choices[i]} {doc["choices"][i]}' for i in range(len(doc['choices']))])
            extracted_answer = match_multiple_choice(f"{doc['question']}\nSelect from the following choices", options, gpt_answer)
            prediction = extracted_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                gpt_answer = gpt_answer.split('\n\n')[-1]
            prediction = intersect[0]
        return prediction, gpt_answer
    except Exception as e:
        print(f"Error: {e}")
        return None, gpt_answer


# Pass in video path here
# Can only work correctly with video llm
def blink_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    visual = [doc[k] for k in ['image_1', 'image_2', 'image_3', 'image_4'] if k in doc and doc[k]]
    return visual


# This is the place where you format your question
def blink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    task_name = doc['sub_task']
    need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']
    disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"
    prompt = doc['prompt']
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{prompt}{post_prompt}"


def blink_doc_to_answer(doc):
    answer = doc["answer"]
    if "(" not in answer:
        answer = f"({answer})"
    return answer


# Process result for mcq answer generation
def blink_process_results_generation(doc, result):
    pred = remove_think(result[0])
    all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(doc['choices'])]
    pred, full_pred = analyze_answer(doc, pred, all_choices)
    pred = pred.strip()
    gt = blink_doc_to_answer(doc)
    score = int(pred == gt)

    return {"full_pred": full_pred, "pred": pred, "ground_truth": gt, "score": score}


def blink_aggregate_score(results, args):
    return np.mean(results)