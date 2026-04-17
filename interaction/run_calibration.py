#!/usr/bin/env python3
"""
Run calibration recordings for participants (paper-mode).

This script assumes physical board configurations (paper) labelled by ids.
For each participant it iterates over config ids (default 1..5) and allows the
operator to trigger recordings interactively:
 - press Enter to record the next config in sequence
 - type a config id (e.g. 3) + Enter to record that config now
 - type 'q' + Enter to finish the participant

Each recorded audio is processed by ImportantFeaturesExtractor and appended to
multimodal_perception/data/calibration_phase/participant_{id}.csv
"""

import argparse
import os
import pandas as pd
from datetime import datetime, timezone

from multimodal_perception.audio.transcribe_audio import WhisperTranscriber
from multimodal_perception.audio.recorder import AudioRecorder
from multimodal_perception.audio.important_feature_extractor import ImportantFeaturesExtractor

CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'multimodal_perception', 'data', 'calibration_phase'))
os.makedirs(CALIB_FOLDER, exist_ok=True)
CALIBRATION_STT_PROMPT = (
    "The speaker says a short single-word English clue for image associations. "
    "Likely words include urban, triangle, puzzle, owl, road, window, fish, night, and shadow."
)


def append_row_to_csv(participant_id, row):
    out_path = os.path.join(CALIB_FOLDER, f"participant_{participant_id}.csv")
    df = pd.DataFrame([row])
    if os.path.exists(out_path):
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Appended features to {out_path}")


def run_for_participant(participant_id, config_ids, device_index=3):
    print(f"\n=== Calibration (paper mode) for participant {participant_id} ===")
    recorder = AudioRecorder(device_index=device_index, channels=2)
    whisper = WhisperTranscriber(initial_prompt=CALIBRATION_STT_PROMPT)
    extractor = ImportantFeaturesExtractor(whisper)

    remaining = list(config_ids)
    print("Interactive paper mode: type a config id (e.g. 1) and press Enter to record that config now.")
    print("Press Enter with no input to record the next config in sequence. Type 'q' to finish this participant.")

    while remaining:
        print(f"Remaining configs: {', '.join(remaining)}")
        choice = input("Config to record (or Enter to take next): ").strip()
        if choice.lower() == 'q':
            print("Finishing participant session.")
            break

        if choice == '':
            cfg = remaining.pop(0)
        elif choice in remaining:
            cfg = choice
            remaining.remove(choice)
        else:
            print("Invalid choice. Enter one of the remaining ids or press Enter.")
            continue

        print(f"\nPlease place configuration #{cfg} (paper) in front of the participant.")
        input("Press Enter when ready to record the clue for this configuration...")

        print("Recording: press Enter to stop")
        recorder.start()
        input()
        audio_path = recorder.stop()

        if audio_path is None:
            print("No audio captured, skipping")
            continue

        print(f"Recorded to {audio_path}. Extracting features...")
        features = extractor.extract(audio_path)

        row = {
            'participant_id': participant_id,
            'config_id': cfg,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **features
        }

        append_row_to_csv(participant_id, row)

        try:
            os.unlink(audio_path)
        except Exception:
            pass

        print("Done. Move to next configuration when ready.")


def main():
    run_for_participant(
        participant_id=0,
        device_index=1,  # Run python -m sounddevice to list devices and find the correct index for your microphone
        config_ids=['1', '2', '3', '4', '5']
    )


if __name__ == '__main__':
    main()
