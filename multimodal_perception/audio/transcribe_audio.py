import whisper
import re

DEFAULT_INITIAL_PROMPT = (
    "This is spoken English in the board game Codenames. "
    "Transcribe exactly what is spoken. "
    "The final clue is usually one single word followed by a number from 1 to 8. "
    "Prefer plain English clue words, for example: 'urban two', 'shell two', 'animal four', 'glass two'. "
    "Do not output non-English text unless it was clearly spoken."
)


class WhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model("small.en")

    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            task="transcribe",
            language="en",
            fp16=False,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=DEFAULT_INITIAL_PROMPT
        )
        transcript = result["text"]
        if self._contains_prompt_leak(transcript):
            result = self.model.transcribe(
                audio_path,
                task="transcribe",
                language="en",
                fp16=False,
                word_timestamps=True,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        transcript = result["text"]
        print("Transcript: {}".format(transcript))

        asr_words = []
        for seg in result['segments']:
            for w in seg['words']:
                asr_words.append({
                    "word": self.clean_asr_word(w['word']),
                    "start": w['start'],
                    "end": w['end']
                })

        return transcript, asr_words

    @staticmethod
    def _contains_prompt_leak(transcript):
        marker = "the final clue is usually one single word followed by a number"
        return marker in transcript.lower()

    @staticmethod
    def clean_asr_word(word):
        word = word.strip()
        word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        word = word.replace('.', '')
        return word
