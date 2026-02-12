import whisper
import re

def transcribe_audio(audio_path):
    _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(
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
                "word": clean_asr_word(w['word']),
                "start": w['start'],
                "end": w['end']
            })

    return transcript, asr_words


def clean_asr_word(word):
    word = word.strip()
    word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
    word = word.replace('.', '')
    return word
