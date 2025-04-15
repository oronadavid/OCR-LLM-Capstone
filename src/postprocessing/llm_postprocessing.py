import re
import json

def strip_markdown(text):
    # Remove code blocks and markdown symbols
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[#*_`]+", "", text)
    return text

def normalize_spacing(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_json(text):
    try:
        json_match = re.search(r'{.*}', text, flags=re.DOTALL)
        return json.loads(json_match.group()) if json_match else {}
    except Exception as e:
        return {"error": f"Failed to extract JSON: {str(e)}"}

def clean_response(
    response: str,
    strip_markdown=True,
    normalize_spacing=True,
    extract_json=False
):
    # Apply steps one at a time based on config
    if strip_markdown:
        response = strip_markdown(response)
    
    if normalize_spacing:
        response = normalize_spacing(response)
    
    if extract_json:
        return extract_json(response)

    return response
