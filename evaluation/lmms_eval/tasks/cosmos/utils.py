import re
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from lmms_eval.cot_utils import remove_think

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
from loguru import logger as eval_logger


def parse_letter_response(output_text: str) -> tuple[str, str]:
    """
    Parses model output text expected to contain a single letter answer (A-Z).

    Extracts the first single uppercase letter found in the text.
    Reasoning is not extracted by this parser.

    Args:
        output_text: The raw text response from the model.

    Returns:
        A tuple containing the extracted answer (str) and an empty reasoning string (str).
        Returns empty strings if no uppercase letter is found.
    """
    answer = ""

    # Look for the first single uppercase letter in the text
    SINGLE_LETTER_PATTERN = re.compile(r"[A-Z]")
    letter_match = SINGLE_LETTER_PATTERN.search(output_text)
    if letter_match:
        answer = letter_match.group(0)

    return answer

# Pass in video path here
# Can only work correctly with video llm
def cosmos_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    subtask = lmms_eval_specific_kwargs["sub_task"]
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    video_path = os.path.join(visual_root, subtask, doc["video"])

    return [video_path]


# This is the place where you format your question
def cosmos_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    qa_pairs = doc["qa_pairs"]
    question = qa_pairs["question"]
    choices = qa_pairs["index2ans"]

    # Format question with choices
    prompt_text = question + "\n"
    prompt_text += "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
    return f"{pre_prompt}{prompt_text}{post_prompt}"


def cosmos_doc_to_answer(doc):
    return doc["qa_pairs"]["answer"].upper()


# Process result for mcq answer generation
def cosmos_process_results_generation(doc, result):
    pred = remove_think(result[0])
    pred = parse_letter_response(pred)
    gt = cosmos_doc_to_answer(doc)
    score = int(pred == gt)

    # return {"score": {"pred": pred, "ground_truth": gt, "score": score, "question_id": question_id, "task": task}}
    return {"pred": pred, "ground_truth": gt, "score": score}


def cosmos_aggregate_score(results, args):
    return np.mean(results)