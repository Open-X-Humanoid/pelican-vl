import re
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from lmms_eval.cot_utils import remove_think


with open(Path(__file__).parent / "where2place.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

def text2pts(text, width=640, height=480):
    simple_pattern = r"[\[\(]([-+]?\d+\.?\d*(?:\s*,\s*[-+]?\d+\.?\d*)*?)[\]\)]"
    nested_pattern = r"\[(?:\[[-+]?\d+\.?\d*(?:\s*,\s*[-+]?\d+\.?\d*)*?\](?:\s*,\s*\[[-+]?\d+\.?\d*(?:\s*,\s*[-+]?\d+\.?\d*)*?\])*?)\]"
    matches = re.findall(simple_pattern, text) + re.findall(nested_pattern, text)
    points = []
    for match in matches:
        vector = [
            float(num) if '.' in num else int(num) for num in match.split(',')
        ]
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    return np.array(points)

# Pass in video path here
# Can only work correctly with video llm
def where2place_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    image_path = os.path.join(visual_root, "images", doc["image"])
    visual = [Image.open(image_path)]
    return visual


# This is the place where you format your question
def where2place_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # Format question with choices
    question = doc["text"]
    return f"{pre_prompt}{question}{post_prompt}"


def where2place_doc_to_answer(doc):
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    image_name = doc["image"]
    mask_path = os.path.join(visual_root, "masks", image_name)
    return mask_path


# Process result for mcq answer generation
def where2place_process_results_generation(doc, result):
    pred = remove_think(result[0])
    mask_path = where2place_doc_to_answer(doc)
    mask = np.array(Image.open(mask_path)) / 255.
    try:
        points = text2pts(pred)
        score = 0
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                       & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            score = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
    except:
        print(f'Failed to parse answer for question: {pred}')
        score = -1

    # return {"score": {"pred": pred, "ground_truth": gt, "score": score, "question_id": question_id, "task": task}}
    return {"pred": pred, "ground_truth": mask_path, "score": score}


def where2place_aggregate_score(results, args):
    valid_results = [x for x in results if x >= 0]
    score = np.mean(valid_results)
    num_valid = len(valid_results)
    num_total = len(results)
    return {"score": score, "valid/total": f"{num_valid}/{num_total}"}