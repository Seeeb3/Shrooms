from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from scipy.stats import uniform

# 1. Load data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)


# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.base import clone

# --- Train and evaluate default model ---
default_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
default_model.fit(X_train, y_train)
y_pred_default = default_model.predict(X_test)
print("Evaluation - Logistic Regression (Default):")
print(classification_report(y_test, y_pred_default, target_names=["poisonous", "edible"]))
joblib.dump(default_model, "models/logistic_regression_default.pkl")

# --- Hyperparameter tuning ---
param_distributions = {
    "C": uniform(0.01, 10),
    "penalty": ["l2"],
    "solver": ["lbfgs", "saga"]
}

search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
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
print("Evaluation - Logistic Regression (Tuned):")
print(classification_report(y_test, y_pred_tuned, target_names=["poisonous", "edible"]))
joblib.dump(tuned_model, "models/logistic_regression_tuned.pkl")