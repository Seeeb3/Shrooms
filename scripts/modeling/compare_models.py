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
    "RandomForest (Default)": joblib.load("models/random_forest_default.pkl"),
    "RandomForest (Tuned)": joblib.load("models/random_forest_tuned.pkl"),
    "SVM (Default)": joblib.load("models/svc_default.pkl"),
    "SVM (Tuned)": joblib.load("models/svc_tuned.pkl"),
    "NB Multinomial (Default) fit_prior=false": joblib.load("models/naive_bayes_multinomialnb_(fit_prior=false)_(default).pkl"),
    "NB Complement (Default) fit_prior=false": joblib.load("models/naive_bayes_complementnb_(fit_prior=false)_(default).pkl"),
    "NB Bernoulli (Default) fit_prior=false": joblib.load("models/naive_bayes_bernoullinb_(fit_prior=false)_(default).pkl"),
    "NB Multinomial (Tuned) fit_prior=false": joblib.load("models/naive_bayes_multinomialnb_(fit_prior=false)_(tuned).pkl"),
    "NB Complement (Tuned) fit_prior=false": joblib.load("models/naive_bayes_complementnb_(fit_prior=false)_(tuned).pkl"),
    "NB Bernoulli (Tuned) fit_prior=false": joblib.load("models/naive_bayes_bernoullinb_(fit_prior=false)_(tuned).pkl"),
    "NB Multinomial (Default) fit_prior=true": joblib.load("models/naive_bayes_multinomialnb_(fit_prior=true)_(default).pkl"),
    "NB Complement (Default) fit_prior=true": joblib.load("models/naive_bayes_complementnb_(fit_prior=true)_(default).pkl"),
    "NB Bernoulli (Default) fit_prior=true": joblib.load("models/naive_bayes_bernoullinb_(fit_prior=true)_(default).pkl"),
    "NB Multinomial (Tuned) fit_prior=true": joblib.load("models/naive_bayes_multinomialnb_(fit_prior=true)_(tuned).pkl"),
    "NB Complement (Tuned) fit_prior=true": joblib.load("models/naive_bayes_complementnb_(fit_prior=true)_(tuned).pkl"),
    "NB Bernoulli (Tuned) fit_prior=true": joblib.load("models/naive_bayes_bernoullinb_(fit_prior=true)_(tuned).pkl"),
    "LogisticRegression (Default)": joblib.load("models/logistic_regression_default.pkl"),
    "LogisticRegression (Tuned)": joblib.load("models/logistic_regression_tuned.pkl"),
    "RidgeClassifier (Default)": joblib.load("models/ridge_classifier_default.pkl"),
    "RidgeClassifier (Tuned)": joblib.load("models/ridge_classifier_tuned.pkl"),
    "PassiveAggressive (Default)": joblib.load("models/passive_aggressive_default.pkl"),
    "PassiveAggressive (Tuned)": joblib.load("models/passive_aggressive_tuned.pkl"),
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

# 7. Plot horizontal bar chart of all models by F1-score
def plot_bar_chart(df, metric="F1-score (macro)", output_path="results/f1_score_comparison.png"):
    models = df["Model"]
    scores = df[metric]

    colors = plt.cm.tab20.colors
    if len(models) > len(colors):
        import itertools
        colors = list(itertools.islice(itertools.cycle(colors), len(models)))

    plt.figure(figsize=(16, min(0.4 * len(models) + 2, 10)))
    bars = plt.barh(models, scores, color=colors[:len(models)])
    plt.xlabel(metric)
    plt.title(f"All Models by {metric}")
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

plot_bar_chart(df_summary)
print("Bar chart saved to results/f1_score_comparison.png")