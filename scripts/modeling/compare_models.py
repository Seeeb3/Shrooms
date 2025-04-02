import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from evaluate import print_metrics, plot_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np

# 1. Load vectorizer and models
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
models = {
    "XGBoost (Default)": joblib.load("models/xgboost_default.pkl"),
    "XGBoost (Tuned)": joblib.load("models/xgboost_tuned.pkl"),
    "Random Forest (Default)": joblib.load("models/random_forest_default.pkl"),
    "Random Forest (Tuned)": joblib.load("models/random_forest_tuned.pkl"),
    "SVM (Default)": joblib.load("models/svm_default.pkl"),
    "SVM (Tuned)": joblib.load("models/svm_tuned.pkl"),
    "Naive Bayes (Default)": joblib.load("models/naive_bayes_default.pkl"),
    "Naive Bayes (Tuned)": joblib.load("models/naive_bayes_tuned.pkl")
}

# 2. Load and vectorize data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"]

X = vectorizer.transform(X_text)
y_encoded = y.map({"edible": 1, "poisonous": 0})

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Compare models
summary = []

for name, model in models.items():
    print(f"\nEvaluating model: {name}")
    y_pred = model.predict(X_test)
    y_test_label = y_test.map({1: "edible", 0: "poisonous"})
    y_pred_label = pd.Series(y_pred).map({1: "edible", 0: "poisonous"})

    # print_metrics(y_test_label, y_pred_label)
    # plot_confusion_matrix(y_test_label, y_pred_label, labels=["edible", "poisonous"], title=f"Confusion Matrix - {name}")

    summary.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_label, y_pred_label),
        "Precision (macro)": precision_score(y_test_label, y_pred_label, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test_label, y_pred_label, average="macro"),
        "F1-score (macro)": f1_score(y_test_label, y_pred_label, average="macro")
    })

# 5. Display summary
print("\nSummary of model performance (sorted by F1-score):\n")
df_summary = pd.DataFrame(summary)
df_summary = df_summary.sort_values(by="F1-score (macro)", ascending=False).reset_index(drop=True)
print(df_summary.round(4))

# 6. Save summary to CSV
os.makedirs("results", exist_ok=True)
df_summary.to_csv("results/model_comparison.csv", index=False)
print("\nSummary saved to results/model_comparison.csv")