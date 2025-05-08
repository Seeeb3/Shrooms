from xgboost import XGBClassifier
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

# 4. Train XGBoost Classifier
# 4.1 Train XGBoost model with class_weight
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=4.35)
print("\nTraining XGBoost Classifier (class_weight)...")
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("\nEvaluation - XGBoost Classifier (class_weight):\n")
print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
joblib.dump(xgb_model, "models/xgboost_balanced.pkl")

# 4.2 Train XGBoost model with SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model_smote = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
print("\nTraining XGBoost Classifier (SMOTE)...")
xgb_model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = xgb_model_smote.predict(X_test)
print("\nEvaluation - XGBoost Classifier (SMOTE):\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(xgb_model_smote, "models/xgboost_SMOTE.pkl")