from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import csv
import json
import time
import base64
import subprocess
import tempfile

load_dotenv("../../config/.env")
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# for m in client.models.list():
#     print(m.name)

model = "gemini-2.5-flash"  # latest with video understanding (as of June 2024)


def get_last_10_seconds(input_path):
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_file.name
    temp_file.close()

    cmd = [
        "ffmpeg",
        "-sseof", "-10",   # last 10 seconds
        "-i", input_path,
        "-c", "copy",
        output_path,
        "-y"
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def evaluate_video(video_path, allow_fallback=True):
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    prompt = """
You are analyzing a person giving a clue in a Codenames game.

Estimate:
- Speaker confidence: low / medium / high
- Clue difficulty: low / medium / high

Respond ONLY in VALID JSON:
{"confidence": "low|medium|high", "difficulty": "low|medium|high"}
"""

    # Try sending a typed Part (preferred). If SDK rejects raw bytes, fall back to base64 string.
    try:
        video_part = types.Part.from_bytes(
            data=video_bytes,
            mime_type="video/mp4"
        )

        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                video_part,
            ],
        )

    except Exception as e:
        # Detect common pydantic validation for media fields (data/filename) and try fallback
        err_str = str(e)
        print("Media send failed:", err_str)

        if allow_fallback:
            # Fallback: send prompt + base64-encoded video as a string (avoids SDK model validation)
            b64 = base64.b64encode(video_bytes).decode("ascii")
            b64_payload = f"data:video/mp4;base64,{b64}"
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        b64_payload,
                    ],
                )
            except Exception as e2:
                print("Fallback media send also failed:", e2)
                # As last resort, call model with prompt only to avoid crash
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt],
                )
        else:
            # Not allowed to fallback, re-raise
            raise

    text = response.text.strip()

    try:
        return json.loads(text)
    except Exception:
        print("Parse error:", text)
        return {"confidence": "error", "difficulty": "error"}


# ==== PROCESS FOLDER ====
video_folder = "../../assets/video/pilot"
output_csv = "predictions.csv"

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["clue_id", "confidence_pred", "difficulty_pred"])

    files = os.listdir(video_folder)
    for file in files:
        if file.endswith(".mp4"):
            path = os.path.join(video_folder, file)

            print(f"Processing {file}...")
            trimmed_path = get_last_10_seconds(path)
            result = evaluate_video(trimmed_path, False)

            # detect clue id from file name clue_0034.mp4 -> 34
            clue_id = os.path.splitext(file)[0].split("_")[-1]
            writer.writerow([
                clue_id,
                result.get("confidence", "error"),
                result.get("difficulty", "error"),
            ])

            time.sleep(10)

print("Saved to predictions.csv")
