import json
import re
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "sentiment_model.onnx"
VOCAB_PATH = "vocab.json"
MAX_LEN = 128

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())

def encode_text(text):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
    ids = ids[:MAX_LEN]
    ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    return ids

@app.get("/")
def root():
    return {
        "service": "sentiment-analysis",
        "status": "running",
        "engine": "ONNX Runtime"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()

    if not text:
        return {"error": "Text cannot be empty"}

    x = np.array([encode_text(text)], dtype=np.int64)

    logits = session.run(
        ["logits"],
        {"input_ids": x}
    )[0][0]

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    label = "positive" if probs[1] >= probs[0] else "negative"

    return {
        "text": text,
        "sentiment": label,
        "positive_probability": float(probs[1]),
        "negative_probability": float(probs[0])
    }
