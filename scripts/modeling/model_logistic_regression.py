from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# 1. Load data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train Logistic Regression model
# 4.1 Train Logistic Regression with class_weight='balanced'
default_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
print("\nTraining Logistic Regression...")
default_model.fit(X_train, y_train)
y_pred_default = default_model.predict(X_test)
print("\nEvaluation - Logistic Regression:\n")
print(classification_report(y_test, y_pred_default, target_names=["poisonous", "edible"]))
joblib.dump(default_model, "models/logistic_regression_balanced.pkl")

# 4.2 Train Logistic Regression with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_model = LogisticRegression(max_iter=1000, random_state=42)
print("\nTraining Logistic Regression with SMOTE...")
smote_model.fit(X_train_smote, y_train_smote)
y_pred_smote = smote_model.predict(X_test)
print("\nEvaluation - Logistic Regression with SMOTE:\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(smote_model, "models/logistic_regression_SMOTE.pkl")