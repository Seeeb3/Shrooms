from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from imblearn.over_sampling import SMOTE
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

# Apply SMOTE oversampling to the training set only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# 4. Train and evaluate Naive Bayes models
models = {
    "naive_bayes_multinomial.pkl": MultinomialNB(),
    "naive_bayes_complement.pkl": ComplementNB(),
    "naive_bayes_bernoulli.pkl": BernoulliNB()
}

# 4.1 Train and evaluate Naive Bayes default
for model_name, model in models.items():
    print(f"\nTraining Naive Bayes without SMOTE: {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nEvaluation - {model.__class__.__name__}:\n")
    print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
    joblib.dump(model, f"models/{model_name.replace('.pkl', '_default.pkl')}")

# 4.2 Train and evaluate Naive Bayes with SMOTE
for model_name, model in models.items():
    print(f"\nTraining Naive Bayes: {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nEvaluation - {model.__class__.__name__}:\n")
    print(classification_report(y_test, y_pred, target_names=["poisonous", "edible"]))
    joblib.dump(model, f"models/{model_name}")