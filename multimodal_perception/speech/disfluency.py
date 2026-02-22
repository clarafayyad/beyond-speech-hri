from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def get_disfluency(transcript):
    model_name = "4i-ai/BERT_disfluency_cls"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = clf(transcript)
    disfluency_score = result[0]["score"]
    return disfluency_score