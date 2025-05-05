from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import loguniform
import pandas as pd
import joblib
import numpy as np

# Load data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train and evaluate default model ---
default_model = PassiveAggressiveClassifier(random_state=42, class_weight="balanced")
default_model.fit(X_train, y_train)
y_pred_default = default_model.predict(X_test)
print("Evaluation - PassiveAggressiveClassifier (Default):")
print(classification_report(y_test, y_pred_default, target_names=["poisonous", "edible"]))
joblib.dump(default_model, "models/passive_aggressive_default.pkl")

# --- Hyperparameter tuning ---
param_distributions = {
    "C": loguniform(1e-4, 10),
    "loss": ["hinge", "squared_hinge"]
}

search = RandomizedSearchCV(
    PassiveAggressiveClassifier(random_state=42, class_weight="balanced"),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring="f1_macro",
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
tuned_model = search.best_estimator_
print("Best parameters:")
print(search.best_params_)
print(f"Best F1-score: {search.best_score_:.4f}")

# --- Evaluate tuned model ---
y_pred_tuned = tuned_model.predict(X_test)
print("Evaluation - PassiveAggressiveClassifier (Tuned):")
print(classification_report(y_test, y_pred_tuned, target_names=["poisonous", "edible"]))
joblib.dump(tuned_model, "models/passive_aggressive_tuned.pkl")
