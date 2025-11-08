import re
import numpy as np
import yaml
from PIL import Image
import io
import base64
from lmms_eval.cot_utils import remove_think


ANSWER_LETTERS = ["A", "B", "C", "D"]


def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = answer.strip()
    return answer


def remove_special_characters(text):
    pattern = r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_multiple_choice(answer):
    # reference: https://github.com/flageval-baai/FlagEvalMM/blob/main/flagevalmm/evaluator/pre_process.py
    answer = strip_answer(answer)
    pattern = r"^([A-Z])\."
    matches = re.match(pattern, answer)
    if matches:
        return matches.group(1)
    key_words = [
        "boxed",
        "Answer:",
        "Answer is",
        "answer is",
        "option is",
        "Correct option",
        "correct option",
        "Answer",
        "answer",
        "故选",
        "选择",
        "正确选项为",
        "答案选",
        "答案为",
        "答案是",
        "因此",
        "答案",
    ]

    for key_word in key_words:
        if key_word in answer:
            answer = answer.split(key_word)[-1]
            break
    answer = remove_special_characters(answer)
    # keep the last line
    answer = answer.split("\n")[-1]
    pattern = r"[A-Z]"
    matches = re.findall(pattern, answer)
    return "".join(matches)



# Pass in video path here
# Can only work correctly with video llm
def emb_spatial_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    image_base64 = doc["image"]
    visual = [Image.open(io.BytesIO(base64.b64decode(image_base64)))]
    return visual


# This is the place where you format your question
def emb_spatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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
    if "answer_options" in doc:
        options_text = "\n".join([f"{ANSWER_LETTERS[i]}.{opt}" for i, opt in enumerate(doc["answer_options"])])
        question += f"\nOptions:\n{options_text}"

    return f"{pre_prompt}{question}{post_prompt}"


def emb_spatial_doc_to_answer(doc):
    return ANSWER_LETTERS[doc["answer"]]


# Process result for mcq answer generation
def emb_spatial_process_results_generation(doc, result):
    pred = remove_think(result[0])
    pred = process_multiple_choice(pred)
    gt = emb_spatial_doc_to_answer(doc)
    return {"pred": pred, "ground_truth": gt, "score": int(pred == gt)}



def emb_spatial_aggregate_score(results, args):
    return np.mean(results)