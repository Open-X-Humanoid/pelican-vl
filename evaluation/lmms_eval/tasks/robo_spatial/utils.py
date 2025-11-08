import pandas as pd
import re
import ast
from lmms_eval.tasks.robo_spatial.pre_process import normalize_string

import numpy as np
import yaml
from lmms_eval.cot_utils import remove_think


def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def evaluate_answer(ground_truth, generated_answer):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()

    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = gt_lower == "yes"
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False  # Default to not parsable until we successfully parse

        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable

            # Extract the first coordinate pair using regex
            # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2] or [0.1,0.2]
            # This approach is more robust than trying to parse the entire list

            # Try to match tuple format (x,y) or (x, y)
            tuple_match = re.search(
                r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", generated_answer
            )
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails

            # Try to match list format [x,y] or [x, y]
            list_match = re.search(
                r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", generated_answer
            )
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails

            # Fall back to the original approach but with extra safety
            try:
                # Extract the first list (text between square brackets) from generated_answer
                # Use a regex that can handle multi-line content
                match = re.search(r"\[(.*?)\]", generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable

                # Add spaces after commas if not present (to help ast.literal_eval)
                list_content = match.group(1)
                list_content = re.sub(r",(\S)", r", \1", list_content)

                # Try to fix truncated tuples by adding closing parenthesis and brackets if needed
                list_content = list_content.strip()
                if list_content.endswith(","):
                    list_content = list_content[:-1]

                list_str = "[" + list_content + "]"

                # Try to parse the list directly
                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # If direct parsing fails, try to extract just the first tuple
                    tuple_match = re.search(
                        r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", list_content
                    )
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable

                # Handle different formats for points
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable

                    # Case 1: The list itself is a point coordinates [x, y]
                    if len(gen_val) == 2 and all(
                        isinstance(v, (int, float)) for v in gen_val
                    ):
                        gen_point = tuple(gen_val)  # Convert [x, y] to (x, y)
                    # Case 2: The list contains points [(x, y), ...]
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    # Case 3: The list contains coordinate pairs as lists [[x, y], ...]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])  # Convert [x, y] to (x, y)
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable

                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                # If all parsing attempts fail, return False
                return False, is_binary, parsed_answer, is_parsable

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable

# Pass in video path here
# Can only work correctly with video llm
def robo_spatial_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    visual = [doc["img"]]
    return visual

def get_post_prompt(doc):
    ## reference: https://github.com/flageval-baai/FlagEvalMM/blob/main/tasks/robo_spatial_home/robo_spatial_home_all.py
    post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

    Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

    Format your final answer strictly as follows on the last line of your response:
    Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

    Do not include additional text after this line.
    """

    post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

    Format your final answer strictly as follows on the last line of your response:
    Answer: yes or no

    Do not include additional text after this line.
    """
    if doc["category"] == "context":
        return post_prompt_point
    else:
        return post_prompt_yes_no

# This is the place where you format your question
def robo_spatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    need_to_replace = [
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points.",
        "Answer yes or no.",
    ]
    question = doc.get("question", "")
    for s in need_to_replace:
        question = question.replace(s, "").strip()
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = get_post_prompt(doc)
    # Format question with choices
    instruction = f"{question}{post_prompt}"

    return instruction


def robo_spatial_doc_to_answer(doc):
    ground_truth = doc.get("answer", "")
    return ground_truth


# Process result for mcq answer generation
def robo_spatial_process_results_generation(doc, result):
    pred = remove_think(result[0])
    pred = normalize_string(pred.strip().split("\n")[-1])
    ground_truth = robo_spatial_doc_to_answer(doc)
    correct, is_binary, parsed_answer, is_parsable = evaluate_answer(ground_truth, pred)
    score = int(correct)

    return {"robo_spatial_score": {"score": score, "category": doc["split"]}, "score": score}

def robo_spatial_aggregate_score(results, args):
    df = pd.DataFrame(results)
    overall_score = df["score"].mean()
    category_score_dict = df.groupby("category")["score"].mean().to_dict()
    return {"overall": overall_score, **category_score_dict}