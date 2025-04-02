import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import randint
from evaluate import print_metrics, plot_confusion_matrix

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

# 4. Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# 5. Train default model
model_default = RandomForestClassifier(class_weight="balanced", random_state=42)
model_default.fit(X_resampled, y_resampled)
joblib.dump(model_default, "models/random_forest_default.pkl")

# 6. Hyperparameter tuning
print("\nStarting randomized hyperparameter tuning for Random Forest...")
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None]
}

search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
search.fit(X_resampled, y_resampled)
model_tuned = search.best_estimator_
joblib.dump(model_tuned, "models/random_forest_tuned.pkl")

print("\nBest parameters:")
print(search.best_params_)
print("\nBest F1-score:", search.best_score_)

# 7. Evaluate both models
for label, model in zip(["Default", "Tuned"], [model_default, model_tuned]):
    print(f"\nEvaluation on 20% test split - Random Forest ({label}):")
    y_pred = model.predict(X_test)
    y_test_label = y_test.map({1: "edible", 0: "poisonous"})
    y_pred_label = pd.Series(y_pred).map({1: "edible", 0: "poisonous"})

    print_metrics(y_test_label, y_pred_label)
    plot_confusion_matrix(y_test_label, y_pred_label, labels=["edible", "poisonous"], title=f"Random Forest ({label})")

    print("\n5-Fold Cross-Validation:")
    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(f"Mean F1-score: {scores.mean():.4f} ± {scores.std():.4f}")

# 8. ROC Curve (Tuned only)
y_score = model_tuned.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Tuned)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
