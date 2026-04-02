#!/usr/bin/env python3
"""
Run calibration recordings for participants.

For each participant id this script iterates over board images (default: ./boards/*.png),
shows the image (macOS `open`), records a spoken clue, extracts audio features using
ImportantFeaturesExtractor (which uses Whisper for transcription), and appends the
features to a per-participant CSV in the calibration folder.

Usage (interactive):
    python interaction/run_calibration.py --participants 01,02

You can also pass --boards to limit which board images to use.
"""

import argparse
import os
import time
import pandas as pd
from datetime import datetime
import subprocess

from multimodal_perception.audio.transcribe_audio import WhisperTranscriber
from multimodal_perception.audio.recorder import AudioRecorder
from multimodal_perception.audio.important_feature_extractor import ImportantFeaturesExtractor

# calibration folder (same as extractor expects)
CALIB_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'multimodal_perception', 'data', 'calibration-phase'))
BOARDS_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'boards'))

os.makedirs(CALIB_FOLDER, exist_ok=True)


def list_boards(patterns=None):
    files = []
    if patterns:
        for p in patterns:
            path = os.path.join(BOARDS_FOLDER, p)
            if os.path.exists(path):
                files.append(path)
    else:
        # default: all png files in BOARDS_FOLDER sorted
        if os.path.isdir(BOARDS_FOLDER):
            for f in sorted(os.listdir(BOARDS_FOLDER)):
                if f.lower().endswith('.png') or f.lower().endswith('.jpg'):
                    files.append(os.path.join(BOARDS_FOLDER, f))
    return files


def show_image(path):
    try:
        # macOS open; this will use the default image viewer
        subprocess.Popen(['open', path])
    except Exception:
        print(f"Open image: {path}")


def append_row_to_csv(participant_id, row):
    out_path = os.path.join(CALIB_FOLDER, f"participant_{participant_id}.csv")
    df = pd.DataFrame([row])
    if os.path.exists(out_path):
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Appended features to {out_path}")


def run_for_participant(participant_id, boards):
    print(f"\n=== Calibration for participant {participant_id} ===")
    # initialize recorder and extractor (whisper is heavy; create once)
    recorder = AudioRecorder()
    whisper = WhisperTranscriber()
    extractor = ImportantFeaturesExtractor(whisper)

    for board_path in boards:
        board_name = os.path.basename(board_path)
        print(f"\nShowing board: {board_name}")
        show_image(board_path)
        input("Press Enter when the participant has seen the board and you're ready to record the clue...")

        print("Recording: press Enter to stop")
        recorder.start()
        input()  # wait for operator to press Enter to stop recording
        audio_path = recorder.stop()

        if audio_path is None:
            print("No audio captured, skipping")
            continue

        print(f"Recorded to {audio_path}. Extracting features...")
        features = extractor.extract(audio_path)

        # build row
        row = {
            'participant_id': participant_id,
            'board': board_name,
            'timestamp': datetime.utcnow().isoformat(),
            **features
        }

        append_row_to_csv(participant_id, row)

        try:
            os.unlink(audio_path)
        except Exception:
            pass

        print("Done. Move to next board when ready.")


def run_for_participant_paper(participant_id, config_ids):
    print(f"\n=== Calibration (paper mode) for participant {participant_id} ===")
    recorder = AudioRecorder()
    whisper = WhisperTranscriber()
    extractor = ImportantFeaturesExtractor(whisper)

    remaining = list(config_ids)
    print("Interactive paper mode: you can type a config id (e.g. 1) and press Enter to record that config now.")
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
            'timestamp': datetime.utcnow().isoformat(),
            **features
        }

        append_row_to_csv(participant_id, row)

        try:
            os.unlink(audio_path)
        except Exception:
            pass

        print("Done. Move to next configuration when ready.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--participants', '-p', required=True, help='Comma-separated list of participant ids (e.g. 01,02)')
    parser.add_argument('--boards', '-b', help='Optional comma-separated list of board filenames (in boards/). If omitted all boards are used.')
    parser.add_argument('--paper', action='store_true', help='Use paper mode (default): iterate config ids.')
    parser.add_argument('--configs', help='Comma-separated config ids to use in paper mode (e.g. 1,2,3). Defaults to 1..5')
    args = parser.parse_args()

    participants = [p.strip() for p in args.participants.split(',') if p.strip()]
    board_patterns = [p.strip() for p in args.boards.split(',')] if args.boards else None

    # determine mode and configs
    paper_mode = args.paper or not bool(board_patterns)
    if args.configs:
        config_ids = [c.strip() for c in args.configs.split(',') if c.strip()]
    else:
        config_ids = [str(i) for i in range(1, 6)]

    if paper_mode:
        for pid in participants:
            run_for_participant_paper(pid, config_ids)
        return

    boards = list_boards(board_patterns)
    if not boards:
        print(f"No board images found in {BOARDS_FOLDER}. Exiting.")
        return

    for pid in participants:
        run_for_participant(pid, boards)


if __name__ == '__main__':
    main()
