import re
import pandas as pd

import numpy as np
import yaml
from lmms_eval.cot_utils import remove_think

def text2pts(text, width=640, height=480):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
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


def json2pts(text: str, width=640, height=480) -> np.ndarray:
    import json
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return np.empty((0, 2), dtype=int)

    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return np.empty((0, 2), dtype=int)

    points = []
    for item in data:
        if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
            y_norm, x_norm = item["point"]
            x = int(x_norm / 1000 * width)
            y = int(y_norm / 1000 * height)
            points.append((x, y))
    return np.array(points)


# Pass in video path here
# Can only work correctly with video llm
def ref_spatial_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    visual = [doc["image"]]
    return visual


# This is the place where you format your question
def ref_spatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # Format question with choices
    prompt = f'{doc["prompt"]} {doc["suffix"]}'
    instruction = f"{pre_prompt}{prompt}{post_prompt}"
    # instruction = f"Locate {doc['object']} in the image and output the point coordinates in JSON format"

    return instruction


def ref_spatial_doc_to_answer(doc):
    return doc["mask"]


# Process result for mcq answer generation
def ref_spatial_process_results_generation(doc, result):
    pred = remove_think(result[0])
    mask = ref_spatial_doc_to_answer(doc)
    mask = np.array(mask) / 255.
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)

    try:
        points = text2pts(pred, mask.shape[1], mask.shape[0])
        acc = 0.0
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                       (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
    except Exception as e:
        acc = -1

    return {"ref_spatial_score": {"score": acc, "category": doc["split"]}, "score": acc}

def ref_spatial_aggregate_score(results, args):
    df = pd.DataFrame(results)
    valid_df = df[df["score"] >= 0]
    overall_score = valid_df["score"].mean()
    category_score_dict = valid_df.groupby("category")["score"].mean().to_dict()
    valid_number = valid_df.shape[0]
    total_number = df.shape[0]
    return {"overall": overall_score, **category_score_dict, "valid/total": f"{valid_number}/{total_number}"}