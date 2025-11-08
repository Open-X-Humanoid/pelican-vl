import re
import pandas as pd
import time
import json
import os
from pathlib import Path

import yaml
from PIL import Image
from openai import OpenAI
from lmms_eval.cot_utils import remove_think



DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Multi-View Image** - a RGB frame depicting a scene from 6 novel views.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.
2. **Multi-View Image** - a RGB frame depicting a scene from 6 novel views.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive  
1. **Image** - a single RGB frame depicting a scene.  
2. **Multi-View Image** - a RGB frame depicting a scene from 6 novel views.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

LLM_JUDGE_SYSTEM_PROMPT = """
You are a judge for QA tests.

The user will provide:
Question: The original question.
Pred: The predicted answer.
GT: The ground truth answer.

You need to judge whether the predicted answer is correct or not; just judge the final answer.
If the predicted answer is correct, respond with "True".
If the predicted answer is incorrect, respond with "False".
"""


SYS_PROMPTS = {
    "none": DEFAULT_SYSTEM_PROMPT,
    "zeroshot_cot": ZERO_SHOT_COT_SYSTEM_PROMPT,
    "manual_cot": MANUAL_COT_SYSTEM_PROMPT,
}


###############################################################################
#                             Response Formatting                             #
###############################################################################

RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D}.
"""

JSON_FORMAT = """
You need to respond with the answer in JSON format:

```json
{
    "analysis": "The analysis of the image and question",
    "answer": "A"
}
```
"""

LLM_FORMAT = """
Your answer must be clear and accurate.
"""

DIRECT_FORMAT = """
Note: You only need to respond with A, B, C, or D without providing any additional information.
"""

FORMAT_PROMPTS = {
    "re": RE_FORMAT,
    "json": JSON_FORMAT,
    "llm": LLM_FORMAT,
    "direct": DIRECT_FORMAT
}

with open(Path(__file__).parent / "omni_spatial.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

def _chat_with_retry(messages, model: str, client, *, tries: int = 10):
    for attempt in range(tries):
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=2000,
            )
            return comp.choices[0].message.content
        except Exception as e:
            if attempt == tries - 1:
                print("[FATAL] OpenAI error", e)
                return "A"  # fallback
            print("[WARN] OpenAI error - retrying", e)
            time.sleep(1 + attempt)

def llm_judge(question: str, pred: str, gt: str, client, *, judge_model: str = "gpt-4.1-nano") -> bool:
    msgs = [
        {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"Question: {question}\nPred: {pred}\nGT: {gt}"}]},
    ]
    res = _chat_with_retry(msgs, judge_model, client)
    return "true" in res.lower()

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

def make_client(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_API_BASE):
    """Construct an OpenAI client (proxy-friendly)."""
    return OpenAI(api_key=api_key, base_url=base_url)

# Pass in video path here
# Can only work correctly with video llm
def omni_spatial_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    raw_id = doc["id"]
    task_type = doc["task_type"]
    visual_root = (Path(__file__).parent / config['visual_root']).resolve()
    image_path = os.path.join(visual_root, task_type, f"{raw_id.split('_')[0]}.png")
    visual = [Image.open(image_path)]
    return visual


# This is the place where you format your question
def omni_spatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["options"]

    prompt_type = lmms_eval_specific_kwargs["prompt_type"]
    eval_type = lmms_eval_specific_kwargs["eval_type"]
    prompt = SYS_PROMPTS[prompt_type] + '\n' + FORMAT_PROMPTS[eval_type] + '\n\n' + question
    for i in range(len(options)):
        prompt += f"\n{chr(65 + i)}. {options[i]}"
    return prompt


def omni_spatial_doc_to_answer(doc):
    return doc["answer"]


# Process result for mcq answer generation
def omni_spatial_process_results_generation(doc, result):
    response = remove_think(result[0])
    eval_type = config["lmms_eval_specific_kwargs"]["default"]["eval_type"]
    gt_letter = chr(65 + doc["answer"])
    if eval_type == "json":
        try:
            cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
            pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
        except Exception:
            pred_letter = "A"
        flag = pred_letter == gt_letter
    elif eval_type == "re":
        PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
        pred_letter = PATTERN.findall(response)[-1] if len(PATTERN.findall(response)) > 0 else "A"
        flag = pred_letter == gt_letter
    elif eval_type == "direct":
        pred_letter = response.strip().upper()[:1]
        flag = pred_letter == gt_letter
    elif eval_type == "llm":
        prompt = omni_spatial_doc_to_text(doc)
        client = make_client()
        flag = llm_judge(question=prompt, pred=response, gt=gt_letter, client=client, judge_model="gpt-4.1-mini")
    else:
        assert False, f"Unknown eval_type: {eval_type}"
    task_type, sub_task_type = doc["task_type"], doc["sub_task_type"]
    return {"omni_spatial_score": {"score": int(flag), "task_type": task_type, "sub_task_type": sub_task_type}, "score": int(flag)}


def omni_spatial_aggregate_score(results, args=None):
    df = pd.DataFrame(results)

    overall_acc = df["score"].mean()

    task_acc_dict = df.groupby("task_type")["score"].mean().to_dict()

    subtask_acc_dict = df.groupby(
        ["task_type", "sub_task_type"]
    )["score"].mean().reset_index().to_dict(orient="records")
    return {
        "overall": overall_acc,
        "task_level": task_acc_dict,
        "subtask_level": subtask_acc_dict
    }