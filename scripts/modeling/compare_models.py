

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Load test data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Compare models
model_dir = "models"
results = []

for model_file in sorted(os.listdir(model_dir)):
    if not model_file.endswith(".pkl") or "vectorizer" in model_file:
        continue
    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["poisonous", "edible"], output_dict=True)
        f1_macro = report["macro avg"]["f1-score"]
        accuracy = report["accuracy"]
        results.append({
            "model": model_file.replace(".pkl", ""),
            "accuracy": round(accuracy, 4),
            "f1_score_macro": round(f1_macro, 4),
        })
        print(f"✅ Evaluated: {model_file}")
    except Exception as e:
        print(f"⚠️ Skipped {model_file}: {e}")

# Display results
results_df = pd.DataFrame(results).sort_values(by="f1_score_macro", ascending=False)
print("\n🔍 Model Comparison (sorted by macro F1):\n")
print(results_df.to_string(index=False))