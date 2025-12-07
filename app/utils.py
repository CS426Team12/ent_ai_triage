import json
import re

def extract_json_from_model_output(text: str):
    try:
        return json.loads(text)
    except:
        # fallback: extract curly braces region
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {"summary": text[:300], "urgency": "unknown"}

        try:
            return json.loads(match.group(0))
        except:
            return {"summary": text[:300], "urgency": "unknown"}
