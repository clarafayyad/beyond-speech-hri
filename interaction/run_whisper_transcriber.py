#!/usr/bin/env python3
import argparse
import os

from multimodal_perception.audio.recorder import AudioRecorder
from multimodal_perception.audio.transcribe_audio import WhisperTranscriber


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe it with WhisperTranscriber."
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Optional path to an existing audio file. If omitted, records from microphone.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Input device index for recording. Run `python -m sounddevice` to list devices.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate (Hz).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of recording channels.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep recorded temporary WAV file instead of deleting it after transcription.",
    )
    return parser


def _record_audio(device_index, sample_rate, channels):
    recorder = AudioRecorder(
        device_index=device_index,
        sample_rate=sample_rate,
        channels=channels,
    )
    print("Press Enter to start recording...")
    input()
    print("Recording... Press Enter to stop.")
    recorder.start()
    input()
    return recorder.stop()


def main(argv=None):
    args = build_parser().parse_args(argv)
    transcriber = WhisperTranscriber()

    should_delete = False
    audio_path = args.audio_path
    if audio_path is None:
        audio_path = _record_audio(args.device_index, args.sample_rate, args.channels)
        if audio_path is None:
            print("No audio captured.")
            return 1
        should_delete = not args.keep_audio
        print(f"Saved recording to: {audio_path}")

    try:
        transcript, asr_words = transcriber.transcribe_audio(audio_path)
        print("\nTranscript:")
        print(transcript)
        print("\nASR words:")
        for word in asr_words:
            print(word)
        return 0
    finally:
        if should_delete and audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    raise SystemExit(main())
