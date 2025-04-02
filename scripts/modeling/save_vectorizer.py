import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load dataset
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]

# Fit vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X_text)

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\nTF-IDF vectorizer saved to models/tfidf_vectorizer.pkl")