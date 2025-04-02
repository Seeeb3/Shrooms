import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from evaluate import print_metrics, plot_confusion_matrix
import matplotlib.pyplot as plt

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

# 4. Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 5. Train default SVM
model_default = LinearSVC(class_weight=class_weight_dict, max_iter=10000)
model_default.fit(X_train, y_train)
joblib.dump(model_default, "models/svm_default.pkl")

# 6. Hyperparameter tuning
print("\n🔍 Starting hyperparameter tuning for SVM...")
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(
    estimator=LinearSVC(class_weight=class_weight_dict, max_iter=10000),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
model_tuned = grid_search.best_estimator_
joblib.dump(model_tuned, "models/svm_tuned.pkl")

print("\nBest parameters:")
print(grid_search.best_params_)
print("\nBest F1-score:", grid_search.best_score_)

# 7. Evaluate both models
for label, model in zip(["Default", "Tuned"], [model_default, model_tuned]):
    print(f"\nEvaluation on 20% test split - SVM ({label}):")
    y_pred = model.predict(X_test)
    y_test_label = y_test.map({1: "edible", 0: "poisonous"})
    y_pred_label = pd.Series(y_pred).map({1: "edible", 0: "poisonous"})

    print_metrics(y_test_label, y_pred_label)
    plot_confusion_matrix(y_test_label, y_pred_label, labels=["edible", "poisonous"], title=f"SVM ({label})")

    print("\n5-Fold Cross-Validation:")
    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(f"Mean F1-score: {scores.mean():.4f} ± {scores.std():.4f}")
