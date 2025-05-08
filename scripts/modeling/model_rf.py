from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
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

# 4. Train Random Forest model
# 4.0 Random Forest default
rf = RandomForestClassifier(random_state=42)
print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nEvaluation - Random Forest:\n")
print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
joblib.dump(rf, "models/random_forest_default.pkl")

# 4.1 Random Forest with class_weight='balanced'
rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
print("\nTraining Random Forest (Balanced)...")
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)
print("\nEvaluation - Random Forest (Balanced):\n")
print(classification_report(y_test, y_pred_balanced, target_names=["poisonous", "edible"]))
joblib.dump(rf_balanced, "models/random_forest_balanced.pkl")

# 4.2 Random Forest with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(random_state=42)
print("\nTraining Random Forest (SMOTE)...")
rf_smote.fit(X_train_sm, y_train_sm)
y_pred_smote = rf_smote.predict(X_test)
print("\nEvaluation - Random Forest (SMOTE):\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(rf_smote, "models/random_forest_SMOTE.pkl")