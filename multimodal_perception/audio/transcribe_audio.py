import whisper
import re


class WhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model("small.en")

    def transcribe_audio(self, audio_path):
        # First pass (no prompt!)
        result = self._run(audio_path, temperature=0.0)
        transcript = result["text"]

        # Detect model artifacts
        if self._has_prompt_leak(transcript) or self._has_degenerate_repetition(transcript):
            result = self._run(audio_path, temperature=0.2)
            transcript = result["text"]

        # Light cleanup (ONLY obvious artifacts)
        transcript = self._remove_prompt_leak(transcript)
        transcript = self._limit_repetition(transcript)

        print("Transcript:", transcript)

        asr_words = []
        for seg in result["segments"]:
            for w in seg["words"]:
                asr_words.append({
                    "word": self.clean_asr_word(w["word"]),
                    "start": w["start"],
                    "end": w["end"]
                })

        return transcript, asr_words

    def _run(self, audio_path, temperature):
        return self.model.transcribe(
            audio_path,
            task="transcribe",
            language="en",
            fp16=False,
            word_timestamps=True,
            temperature=temperature,
            best_of=5 if temperature > 0 else None,
            beam_size=5 if temperature > 0 else None,
            condition_on_previous_text=False,
        )

    # --- Artifact detection ---

    @staticmethod
    def _has_prompt_leak(text):
        text = text.lower()

        suspicious_phrases = [
            "this is spoken english",
            "transcribe exactly",
            "final clue",
            "board game codenames"
        ]

        return any(p in text for p in suspicious_phrases)

    @staticmethod
    def _has_degenerate_repetition(text):
        words = text.lower().split()

        if len(words) < 15:
            return False

        # detect unnatural repetition
        for n in range(1, 4):
            chunks = [' '.join(words[i:i + n]) for i in range(len(words) - n)]
            counts = {}

            for c in chunks:
                counts[c] = counts.get(c, 0) + 1

                # threshold: way beyond human repetition
                if counts[c] > 8:
                    return True

        return False

    # --- Cleanup (very conservative) ---

    @staticmethod
    def _remove_prompt_leak(text):
        patterns = [
            r"this is spoken english.*?codenames\.?",
            r"transcribe exactly.*?\.",
            r"the final clue.*?\."
        ]

        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        return text.strip()

    @staticmethod
    def _limit_repetition(text):
        words = text.split()
        result = []

        for w in words:
            # allow natural repetition, block pathological loops
            if len(result) >= 4 and all(x == w for x in result[-4:]):
                continue
            result.append(w)

        return " ".join(result)

    @staticmethod
    def clean_asr_word(word):
        word = word.strip()
        word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        word = word.replace('.', '')
        return word
