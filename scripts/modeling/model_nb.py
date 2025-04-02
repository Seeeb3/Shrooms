import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluate import print_metrics, plot_confusion_matrix
import os

# 1. Load dataset
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer and transform
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train default Naive Bayes
model_default = MultinomialNB()
model_default.fit(X_train, y_train)

# 5. Grid search for tuning
param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(
    MultinomialNB(), param_grid=param_grid, scoring="f1_macro", cv=5, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)
model_tuned = grid_search.best_estimator_

print("\nBest parameters for Naive Bayes:")
print(grid_search.best_params_)
print("\nBest F1-score:", grid_search.best_score_)

# 6. Save both models
joblib.dump(model_default, "models/naive_bayes_default.pkl")
joblib.dump(model_tuned, "models/naive_bayes_tuned.pkl")
print("\nModels saved to models/naive_bayes_default.pkl and naive_bayes_tuned.pkl")

# 7. Evaluate both
for label, model in zip(["Default", "Tuned"], [model_default, model_tuned]):
    print(f"\nEvaluation on 20% test split - Naive Bayes ({label}):")
    y_pred = model.predict(X_test)
    y_test_label = y_test.map({1: "edible", 0: "poisonous"})
    y_pred_label = pd.Series(y_pred).map({1: "edible", 0: "poisonous"})

    print_metrics(y_test_label, y_pred_label)
    plot_confusion_matrix(y_test_label, y_pred_label, labels=["edible", "poisonous"], title=f"Naive Bayes ({label})")

    print("\n5-Fold Cross-Validation:")
    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(f"Mean F1-score: {scores.mean():.4f} ± {scores.std():.4f}")