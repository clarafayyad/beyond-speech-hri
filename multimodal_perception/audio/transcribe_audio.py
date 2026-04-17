import whisper
import re


class WhisperTranscriber:
    def __init__(self, model_name="base", initial_prompt=None):
        self.model = whisper.load_model(model_name)
        self.initial_prompt = initial_prompt

    def transcribe_audio(self, audio_path, prompt=None):
        effective_prompt = prompt if prompt is not None else self.initial_prompt
        transcribe_kwargs = {
            "language": "en",
            "fp16": False,
            "word_timestamps": True,
        }
        if effective_prompt:
            transcribe_kwargs["initial_prompt"] = effective_prompt

        result = self.model.transcribe(audio_path, **transcribe_kwargs)
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
