from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train Ridge Classifier
# 4.0 Ridge Classifier default
ridge = RidgeClassifier()
print("\nTraining RidgeClassifier...")
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print("\nEvaluation - RidgeClassifier:\n")
print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
joblib.dump(ridge, "models/ridge_classifier_default.pkl")

# 4.1 Ridge Classifier with class_weight="balanced"
ridge_balanced = RidgeClassifier(class_weight="balanced")
print("\nTraining RidgeClassifier with class_weight='balanced'...")
ridge_balanced.fit(X_train, y_train)
y_pred_balanced = ridge_balanced.predict(X_test)
print("\nEvaluation - RidgeClassifier with class_weight='balanced':\n")
print(classification_report(y_test, y_pred_balanced, target_names=["poisonous", "edible"]))
joblib.dump(ridge_balanced, "models/ridge_classifier_balanced.pkl")

# 4.2 Ridge Classifier with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
ridge_smote = RidgeClassifier()
print("\nTraining RidgeClassifier with SMOTE oversampling...")
ridge_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = ridge_smote.predict(X_test)
print("\nEvaluation - RidgeClassifier with SMOTE:\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(ridge_smote, "models/ridge_classifier_SMOTE.pkl")