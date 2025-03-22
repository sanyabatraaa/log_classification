import joblib
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer('all-MiniLm-L6-v2')
model_classification = joblib.load("models/log_classifier.joblib")

def classify_with_bert(log_message):
    embeddings = model_embedding.encode([log_message])
    probabilities = model_classification.predict_proba(embeddings)[0]
    if max(probabilities)<0.5:
        return "Unclassified"

    predicted_label = model_classification.predict(embeddings)[0]
    return predicted_label