# analyze.py
from google import genai
from google.genai import types
from PIL import Image
import io, os, json

# Read the API key from env var 
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it in your shell"
    )

client = genai.Client(api_key=api_key)

def _mime_from_pillow_format(fmt: str) -> str:
    fmt = (fmt or "JPEG").upper()
    return {
        "JPEG": "image/jpeg",
        "JPG":  "image/jpeg",
        "PNG":  "image/png",
        "WEBP": "image/webp",
        "HEIC": "image/heic",
        "HEIF": "image/heif",
    }.get(fmt, "image/jpeg")

def get_llm_response(image_data: bytes) -> dict:
    # Detect the image format to set a correct MIME type
    img = Image.open(io.BytesIO(image_data))
    mime = _mime_from_pillow_format(img.format)

    image_part = types.Part.from_bytes(
        data=image_data,
        mime_type=mime,   
    )

    # Ask for structured JSON
    prompt = (
        "Return a concise analysis of the image as JSON with exactly these fields:\n"
        '{ "caption": string, '
        '"objects": [{"label": string, "count": integer}], '
        '"safety_notes": string }\n'
        "Rules: Don't invent details. If unsure, use 'unknown'. "
        "Limit 'objects' to at most 5 categories with integer counts."
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "caption": {"type": "STRING"},
            "objects": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "count": {"type": "INTEGER"},
                    },
                    "required": ["label", "count"],
                    "propertyOrdering": ["label", "count"],
                },
            },
            "safety_notes": {"type": "STRING"},
        },
        "required": ["caption", "objects"],
        "propertyOrdering": ["caption", "objects", "safety_notes"],
    }

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    # The SDK returns JSON as a string 
    try:
        return json.loads(resp.text)
    except Exception:
        return {"caption": resp.text, "objects": [], "safety_notes": ""}
