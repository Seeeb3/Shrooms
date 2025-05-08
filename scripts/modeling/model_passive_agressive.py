from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE  # Added

# 1. Load data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train PassiveAggressive model
# 4.1 Train PassiveAggressive with class_weight='balanced'
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, class_weight="balanced")
print("\nTraining PassiveAggressiveClassifier...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nEvaluation - PassiveAggressiveClassifier:\n")
print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
joblib.dump(model, "models/passive_aggressive_balanced.pkl")

# 4.2 Train PassiveAggressive with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
print("\nTraining PassiveAggressiveClassifier with SMOTE...")
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)
print("\nEvaluation - PassiveAggressiveClassifier with SMOTE:\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(model_smote, "models/passive_aggressive_SMOTE.pkl")