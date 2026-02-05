import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../config/.env")
api_key = os.environ["OPENAI_API_KEY"]

MAPS_DIR = "maps"
OUTPUT_FILE = "parsed_maps_gpt.json"
MODEL = "gpt-4o"  # vision-capable GPT

client = OpenAI()


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# @TODO: enhance prompt, because configurations are not being correctly generated !
# Prompt to instruct the model
PROMPT = """
You are given an image of a 4x5 Codenames-style board. 
Each card has one of these categories: "blue", "red", "neutral", "assassin".

Rules:
- The board is indexed row-wise from 0 (top-left) to 19 (bottom-right).
- Identify the type of each card and return a JSON object with:
  - "blue": list of indices of blue cards
  - "red": list of indices of red cards
  - "neutral": list of indices of neutral cards
  - "assassin": a single index of the assassin card
- DO NOT guess any other information.
- DO NOT add text, explanation, or code fences.
- Return ONLY valid JSON.

Check carefully: there must be exactly 20 indices, no duplicates, and exactly 1 assassin.
Return ONLY JSON, no text, no markdown, no code fences.
"""

results = {}

for filename in sorted(os.listdir(MAPS_DIR)):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(MAPS_DIR, filename)
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
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip fences if any
    raw_text = raw_text.strip("` \n")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON from model for {filename}:\n{raw_text}")

    results[filename] = parsed

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nAll maps parsed and saved to {OUTPUT_FILE}")
