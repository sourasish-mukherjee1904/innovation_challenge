import json
import re


def extract_json(text):
    """
    Extract the first valid JSON object from the LLM response.
    Returns a Python dictionary if successful, otherwise None.
    """

    # Remove Markdown code fences if present
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Find the first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None