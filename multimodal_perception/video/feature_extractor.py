import cv2
import numpy as np
from fer import FER
import os
import pandas as pd
from tqdm import tqdm

# === Paths ===
VIDEO_FOLDER = "../../assets/video/pilot"
INPUT_CSV = "../data/pilot.csv"
OUTPUT_CSV = "../data/video_features.csv"

# Load face detector (built-in with OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# FER emotion detector
emotion_detector = FER(mtcnn=False)


def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    processed_frames = 0

    neutral_count = 0
    positive_count = 0
    negative_count = 0

    movement = 0.0
    prev_center = None

    look_away_count = 0
    baseline_x = None

    last_emotion = None  # reuse between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 Skip frames (process every 5th frame)
        if frame_count % 5 != 0:
            continue

        # 🔥 Resize (faster processing)
        frame = cv2.resize(frame, (640, 480))
        processed_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        # largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]

        face_img = frame[y:y+h, x:x+w]

        # --- Emotion (run less often) ---
        if frame_count % 10 == 0:
            emotions = emotion_detector.detect_emotions(face_img)
            if emotions:
                emo = emotions[0]["emotions"]
                last_emotion = max(emo, key=emo.get)

        # use last known emotion
        if last_emotion:
            if last_emotion == "neutral":
                neutral_count += 1
            elif last_emotion == "happy":
                positive_count += 1
            elif last_emotion in ["sad", "angry", "fear", "disgust"]:
                negative_count += 1

        # --- Head movement ---
        center = np.array([x + w/2, y + h/2])

        if prev_center is not None:
            movement += np.linalg.norm(center - prev_center)

        prev_center = center

        # --- Look-away proxy ---
        if baseline_x is None:
            baseline_x = center[0]

        if abs(center[0] - baseline_x) > w * 0.25:
            look_away_count += 1

    cap.release()

    if processed_frames == 0:
        return None

    return {
        "neutral_ratio": neutral_count / processed_frames,
        "positive_ratio": positive_count / processed_frames,
        "negative_ratio": negative_count / processed_frames,
        "mean_head_movement": movement / processed_frames,
        "look_away_ratio": look_away_count / processed_frames,
    }


def process_clue(row):
    clue_id = int(row["clue_id"])

    video_path = os.path.join(VIDEO_FOLDER, f"clue_{clue_id:04d}.mp4")

    features = extract_video_features(video_path)

    if features is None:
        raise ValueError("No frames processed")

    return {
        "clue_id": clue_id,
        "confidence": row["Confidence"],
        "difficulty": row["Difficulty"],
        **features
    }


def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            features = process_clue(row)
            rows.append(features)
            print("processed clue", row["clue_id"])

        except Exception as e:
            print("error with clue", row["clue_id"], e)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("saved features to", OUTPUT_CSV)


if __name__ == "__main__":
    main()