import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../config/.env")
api_key = os.environ["OPENAI_API_KEY"]

IMAGE_DIR = "cards"
OUTPUT_FILE = "card_descriptions.json"
MODEL = "gpt-4o"  # vision-capable model

client = OpenAI()


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


PROMPT = """
You are analyzing an image for a word-association board game (similar to Codenames).

Your task is to extract a rich, explicit description that a text-only reasoning agent can later use to generate clues.

Rules:
- Be literal and concrete.
- Do NOT guess emotions or intent unless visually obvious.
- Do NOT name a single “answer word” or label for the image.
- Avoid metaphors and poetic language.
- Focus on objects, relationships, and visual evidence.
- Associations should be general concepts, not specific words.

Return ONLY raw JSON.
Do not use markdown, code blocks, or commentary.
Return JSON in the following format ONLY:

{
  "objects": [],
  "actions_relationships": [],
  "setting_environment": "",
  "attributes": [],
  "associations": []
}
"""

results = {}

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    image_b64 = encode_image(path)

    print(f"Processing {filename}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model returned invalid JSON for {filename}:\n{raw_text}")

    results[filename] = parsed

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved structured descriptions to {OUTPUT_FILE}")
