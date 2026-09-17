# SMS Spam Detector - Render Deployment

Files:
- app.py: Flask API + web UI
- templates/index.html: browser interface
- requirements.txt: Python dependencies
- model_bundle.joblib: trained model + both fitted TF-IDF vectorizers

## Expected bundle keys
The model file must be created from the training notebook with:
model, word_tfidf, char_tfidf

## Render settings
Build Command:
    pip install -r requirements.txt

Start Command:
    gunicorn --bind 0.0.0.0:$PORT app:app
