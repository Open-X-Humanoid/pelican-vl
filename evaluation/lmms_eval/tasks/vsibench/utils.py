import os
from functools import partial
import yaml


import numpy as np
import pandas as pd
from loguru import logger as eval_logger
from word2number import w2n
from lmms_eval.cot_utils import remove_think
from pathlib import Path

with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


def vsibench_doc_to_visual(doc):
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    video_path = os.path.join(visual_root, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

def fuzzy_matching(pred):
    return pred.split()[0].rstrip(".").strip() if len(pred) > 0 else ""


def number_fuzzy_matching(pred: str) -> str:
    if not pred:
        return ""
    import re
    m = re.search(r"[+-]?\d+(?:\.\d+)?", pred)
    return m.group(0) if m else pred

def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        return float(pred)
    except (ValueError, TypeError):
        try:
            return float(w2n.word_to_num(str(pred)))
        except Exception:
            return None

def vsibench_process_results(doc, results):
    pred = remove_think(results[0])
    doc["prediction"] = pred
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            try:
                doc[key] = eval(value)(fuzzy_matching(pred), doc["ground_truth"])
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(number_fuzzy_matching(pred)), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc, "score": doc[key]}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == "success_rate":
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    keys = [
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    ]
    values = [output.pop(k, None) for k in keys]
    valid_values = [v for v in values if v is not None]
    if valid_values:
        output["object_rel_direction_accuracy"] = sum(valid_values) / len(valid_values)
        output["overall"] = sum([_ for _ in output.values()]) / len(output)
        eval_logger.info(f"Evaluation results: {output}")
    else:
        output["overall"] = "-1"
    return output