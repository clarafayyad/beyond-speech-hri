from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class DisfluencyDetector:
    def __init__(self):
        model_name = "4i-ai/BERT_disfluency_cls"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def get_disfluency(self, transcript):
        result = self.clf(transcript)
        disfluency_score = result[0]["score"]
        return disfluency_score
