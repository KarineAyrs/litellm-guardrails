"""Run once (at Docker build time) to download model weights into the image."""
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

models = [
    (
        "cointegrated/rubert-tiny-toxicity",
        os.getenv("TOXICITY_MODEL_PATH", "./models/rubert-tiny-toxicity"),
        AutoModelForSequenceClassification,
    ),
    (
        "Davlan/bert-base-multilingual-cased-ner-hrl",
        os.getenv("NER_MODEL_PATH", "./models/bert-multilingual-ner"),
        AutoModelForTokenClassification,
    ),
    (
        "gbv/mdeberta-ru-prompt-injection",
        os.getenv("PROMPT_INJECTION_MODEL_PATH", "./models/mdeberta-prompt-injection"),
        AutoModelForSequenceClassification,
    ),
]

for model_name, save_path, model_cls in models:
    print(f"Downloading {model_name} -> {save_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    model = model_cls.from_pretrained(model_name)
    model.save_pretrained(save_path)

print("Done.")
