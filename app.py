import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model_bundle.joblib"

bundle = joblib.load(BUNDLE_PATH)

model = bundle["model"]
word_tfidf = bundle["word_tfidf"]
char_tfidf = bundle["char_tfidf"]

app = Flask(__name__)

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_message(message):
    cleaned = [clean_text(message)]

    word_features = word_tfidf.transform(cleaned)
    char_features = char_tfidf.transform(cleaned)

    features = hstack([word_features, char_features]).tocsr()

    prediction = int(model.predict(features)[0])
    spam_probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": "Spam" if prediction == 1 else "Ham",
        "spam_probability": round(spam_probability * 100, 2),
    }

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    return jsonify(predict_message(message))

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
