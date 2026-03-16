import whisper
import re


class WhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            word_timestamps=True
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
