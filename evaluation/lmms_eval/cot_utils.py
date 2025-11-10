import re
def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval

def extract_option_final_answer(text):
    patterns = [
        r"Final answer\s*[:：]?\s*([A-Z])\b",                 # Final answer: A / Final answer:A
        r"Answer\s+is\s*[:：]?\s*([A-Z])\b",                  # Answer is: B
        r"The correct option is\s*([A-Z])\b",                # correct option is F
        r"So the answer is\s*([A-Z])\b",                     # so the answer is G
        r"Option\s*([A-Z])\b",                               # Option J
        r"^\s*([A-Z])\s*[:\-]",                              # H: some reason
        r"^\s*([A-Z])\s*$",                                  # only line is "K"
        r"=>\s*([A-Z])\b",                                   # => M
        r"\(([A-Z])\)",                                      # (N)
        r"\[([A-Z])\]",                                      # [Q]
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None

def extract_box_answer(text, is_strict=True):
    boxed_answer = last_boxed_only_string(text)
    if boxed_answer:
        try:
            boxed_answer = boxed_answer.replace("\\boxed{", "").replace("}", "").replace('\\text{','').replace('}','')
            return boxed_answer
        except:
            return ""
    return "" if is_strict else text

def extract_option_pred(content: str, is_strict=True):
    boxed_answer = last_boxed_only_string(content)
    if boxed_answer:
        try:
            boxed_answer = boxed_answer.replace("\\boxed{", "").replace("}", "").replace('\\text{','').replace('}','')
            if boxed_answer.isalpha():
                boxed_answer = boxed_answer.upper()
            return boxed_answer.strip()
        except:
            return None
    elif not is_strict:
        final_answer = extract_option_final_answer(content)
        if final_answer:
            return final_answer.strip().upper()
    return None

def remove_think(content: str) -> str:
    content = re.sub(r'^.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'^.*?</thought>', '', content, flags=re.DOTALL)
    match = re.search(r'<answer>(.*?)</answer>', content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    content = extract_box_answer(content, is_strict=False)
    return content.strip()

def test_remove_think():
    text = """
<thought>**Determining the Next Action in the Cooking Cleanup Sequence**
Okay, so I've got a video and a still image to work with. The task is to identify the next logical step in cleaning a pan and spatula, and it's pretty straightforward, given the context. My first thought is to break down the problem methodically.
First, I need to understand what the user wants: the *next* action. So, let's look at the available evidence. The image shows hands over a sink with a dirty pan and spatula. The video demonstrates the entire cleaning process, step-by-step. From my experience with these kinds of tasks, this seems like a simple sequential analysis.
The video starts with placing the dirty items in the sink, exactly matching the image. This means I'm looking for the immediate next step after that initial setup. I then watch the video and take mental notes of each action. The hand reaches for the tap in the image, and then in the video the person turns on the water to rinse the items.
Now, let's look at the options. Putting the spatula away is far down the line in this case. Scrubbing comes after rinsing, of course, both the pan and spatula. So, turning on the tap is the most logical next action, as it's the very next thing the person does after placing the dirty items in the sink. The image confirms that intent too.
</thought>A
    """.strip()
    assert remove_think(text) == "A"

    text = "A: Beijing"
    assert remove_think(text) == text

def test_extract_option_pred():
    content = "I love you \\boxed{A}."
    extracted = extract_option_pred(content)
    assert extracted == "A"

    content = "A"
    extracted = extract_option_pred(content)
    assert extracted == None

    content = "A"
    extracted = extract_option_pred(content, is_strict=False)
    assert extracted == "A"

if __name__ == "__main__":
    test_remove_think()
    test_extract_option_pred()