import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from evaluate import print_metrics, plot_confusion_matrix
import joblib
import os

# 1. Load dataset
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train default model
default_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
default_model.fit(X_train, y_train)
joblib.dump(default_model, "models/xgboost_default.pkl")

# 5. Hyperparameter tuning
print("\nRandomized hyperparameter tuning...")
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}

search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0),
    param_distributions=param_dist,
    n_iter=100,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train)
model_tuned = search.best_estimator_
joblib.dump(model_tuned, "models/xgboost_tuned.pkl")

print("\nBest parameters:")
print(search.best_params_)
print("\nBest F1-score:", search.best_score_)

# 6. Evaluate both models
for label, model in zip(["Default", "Tuned"], [default_model, model_tuned]):
    print(f"\nEvaluation on 20% test split - XGBoost ({label}):")
    y_pred = model.predict(X_test)
    y_test_label = y_test.map({1: "edible", 0: "poisonous"})
    y_pred_label = pd.Series(y_pred).map({1: "edible", 0: "poisonous"})

    print_metrics(y_test_label, y_pred_label)
    plot_confusion_matrix(y_test_label, y_pred_label, labels=["edible", "poisonous"], title=f"XGBoost ({label})")

    print("\n5-Fold Cross-Validation:")
    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(f"Mean F1-score: {scores.mean():.4f} ± {scores.std():.4f}")

feature_names = np.array(vectorizer.get_feature_names_out())
importances = model_tuned.feature_importances_
top_indices = importances.argsort()[::-1][:10]

print("\nTop 10 most important words:")
for i in top_indices:
    print(f"{feature_names[i]:<20} → importance: {importances[i]:.4f}")

# 8. ROC Curve (tuned only)
y_score = model_tuned.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost (Tuned)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()