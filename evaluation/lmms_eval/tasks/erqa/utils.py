import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from lmms_eval.cot_utils import remove_think


with open(Path(__file__).parent / "erqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

# Pass in video path here
# Can only work correctly with video llm
def erqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # visual_root = config["visual_root"]
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    print("visual root start")
    print(config['visual_root'])
    print(visual_root)
    print("visual root end")
    visual = [Image.open(os.path.join(visual_root, image_path)) for image_path in doc["images"]]
    return visual


# This is the place where you format your question
def erqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # Format question with choices
    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def erqa_doc_to_answer(doc):
    return doc["answer"]


# Process result for mcq answer generation
def erqa_process_results_generation(doc, result):
    pred = remove_think(result[0])
    gt = erqa_doc_to_answer(doc)
    norm_pred = pred.replace(".", "").strip().lower()
    if len(pred) > 0:
        norm_pred = norm_pred[0]
    score = int(norm_pred == gt.strip().lower())
    return {"pred": pred, "ground_truth": gt, "score": score}


def erqa_aggregate_score(results, args):
    return np.mean(results)