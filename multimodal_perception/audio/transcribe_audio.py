import whisper
import re

DEFAULT_INITIAL_PROMPT = (
    "This is a Codenames spymaster speaking. "
    "Transcribe everything exactly as spoken, including filler words like 'um' or 'uh', hesitations, and corrections. "
    "The speaker may think out loud before giving the final clue. "
    "The final clue is always at the end and consists of one single word followed by a number. "
    "Return the full transcription."
)


class WhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            word_timestamps=True,
            initial_prompt=DEFAULT_INITIAL_PROMPT
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
    def clean_asr_word(word):
        word = word.strip()
        word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        word = word.replace('.', '')
        return word
