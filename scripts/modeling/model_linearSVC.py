from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
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

# 4. Train LinearSVC
# 4.1 LinearSVC with class_weight="balanced"
svc_balanced = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
print("\nTraining LinearSVC with class_weight='balanced'...")
svc_balanced.fit(X_train, y_train)
y_pred_balanced = svc_balanced.predict(X_test)
print("\nEvaluation - LinearSVC with class_weight='balanced':\n")
print(classification_report(y_test, y_pred_balanced, target_names=["poisonous", "edible"]))
joblib.dump(svc_balanced, "models/linear_SVC_balanced.pkl")

# 4.2 LinearSVC with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
svc_smote = LinearSVC(random_state=42, max_iter=10000)
print("\nTraining LinearSVC with SMOTE oversampling...")
svc_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = svc_smote.predict(X_test)
print("\nEvaluation - LinearSVC with SMOTE:\n")
print(classification_report(y_test, y_pred_smote, target_names=["poisonous", "edible"]))
joblib.dump(svc_smote, "models/linear_SVC_SMOTE.pkl")
