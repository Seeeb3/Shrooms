import os
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Load test data
df = pd.read_csv("data/features/mushrooms_dataset.csv")
X_text = df["description"]
y = df["edibility_binary"].map({"edible": 1, "poisonous": 0})

# 2. Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(X_text)

# 3. Train-test split (including original text split)
X_text_train, X_text_test, X_train, X_test, y_train, y_test = train_test_split(
    X_text, X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Compare models
# 4.0 Load all models and evaluate
model_dir = "models"
results = []

for model_file in sorted(os.listdir(model_dir)):
    if not model_file.endswith(".pkl") or "vectorizer" in model_file:
        continue
    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["poisonous", "edible"], output_dict=True)
        results.append({
            "model": model_file.replace(".pkl", ""),
            "accuracy": round(report["accuracy"], 4),
            "precision_macro": round(report["macro avg"]["precision"], 4),
            "recall_macro": round(report["macro avg"]["recall"], 4),
            "f1_score_macro": round(report["macro avg"]["f1-score"], 4),
        })
        print(f"Evaluated: {model_file}")
    except Exception as e:
        print(f"Skipped {model_file}: {e}")

# Display results
results_df = pd.DataFrame(results).sort_values(
    by=["f1_score_macro", "recall_macro", "precision_macro", "accuracy"],
    ascending=[False, False, False, False]
)

print("\nModel Comparison (sorted by macro F1-score):\n")
print(results_df.to_string(index=False))
# Export to CSV
results_df.to_csv("results/model_comparison_summary.csv", index=False)
print("\nResults exported to results/model_comparison_summary.csv\n")

# 4.1 Compare SMOTE and balanced to default for each base model
comparison_rows = []
for base in ["logistic_regression", "linear_SVC", "ridge_classifier", "passive_aggressive", "xgboost", "naive_bayes_complement", "naive_bayes_multinomial", "naive_bayes_bernoulli", "random_forest"]:
    try:
        default = results_df[results_df["model"] == f"{base}_default"].iloc[0]
        for mode in ["SMOTE", "balanced"]:
            variant = results_df[results_df["model"] == f"{base}_{mode}"]
            if not variant.empty:
                delta = variant.iloc[0]["f1_score_macro"] - default["f1_score_macro"]
                comparison_rows.append({
                    "base_model": base,
                    "variant": mode,
                    "f1_score_default": default["f1_score_macro"],
                    "f1_score_variant": variant.iloc[0]["f1_score_macro"],
                    "delta_f1": round(delta, 4)
                })
                # Enregistrement temporaire des F1 par base_model pour comparaison
                if "f1_by_mode" not in locals():
                    f1_by_mode = {}
                if base not in f1_by_mode:
                    f1_by_mode[base] = {}
                f1_by_mode[base][mode] = variant.iloc[0]["f1_score_macro"]
    except IndexError:
        continue

if comparison_rows:
    comp_df = pd.DataFrame(comparison_rows)
    best_modes = []
    for row in comp_df.itertuples():
        base = row.base_model
        f1_smote = f1_by_mode.get(base, {}).get("SMOTE")
        f1_balanced = f1_by_mode.get(base, {}).get("balanced")
        if f1_smote is not None and f1_balanced is not None:
            if f1_smote > f1_balanced:
                best = "SMOTE"
            elif f1_balanced > f1_smote:
                best = "balanced"
            else:
                best = "equal"
        elif f1_smote is not None:
            best = "SMOTE"
        elif f1_balanced is not None:
            best = "balanced"
        else:
            best = "unknown"
        best_modes.append(best)
    comp_df["best_mode"] = best_modes
    print("\nComparison of SMOTE and balanced vs. default (delta F1):\n")
    print(comp_df.to_string(index=False))
    comp_df.to_csv("results/f1_gain_vs_default.csv", index=False)
    print("\nSaved comparison to results/f1_gain_vs_default.csv\n")

# 5. Visualize results
# 5.1 Bar chart visualization
melted_df = results_df.melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_score_macro"],
                            var_name="metric", value_name="score")
fig = px.bar(melted_df, x="model", y="score", color="metric", barmode="group",
             title="Model Performance Comparison", height=600)
fig.update_layout(xaxis_tickangle=-45)
fig.show()
# Save the bar chart as an image
fig.write_html("results/models_comparison_bar_chart.html")
print("\nBar chart saved as results/models_comparison_bar_chart.html\n")
# Save the bar chart as an image
fig.write_image("results/models_comparison_bar_chart.png")
print("\nBar chart saved as results/models_comparison_bar_chart.png\n")

# 5.2 Heatmap visualization
heatmap_df = results_df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_score_macro"]]
fig_heatmap = px.imshow(heatmap_df, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues", title="Model Performance Heatmap")
fig_heatmap.update_layout(yaxis_title="Model", xaxis_title="Metric")
fig_heatmap.show()
# Save the heatmap as an image
fig_heatmap.write_html("results/models_comparison_metrics_heatmap.html")
print("\nHeatmap saved as results/models_comparison_metrics_heatmap.html\n")
# Save the heatmap as an image
fig_heatmap.write_image("results/models_comparison_metrics_heatmap.png")
print("\nHeatmap saved as results/models_comparison_metrics_heatmap.png\n")

# 6. Top 10 SHAP features from the top 3 models
top_models = ["xgboost_SMOTE", "linear_SVC_SMOTE", "passive_aggressive_SMOTE"]
shap_values_dict = {}
top_shap_features = {}
print("\nTop 10 SHAP features for the 3 best models:\n")

for model_name in top_models:
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    model = joblib.load(model_path)
    # Handle sparse input compatibility for SHAP
    if hasattr(model, "coef_") or model_name == "xgboost_SMOTE":
        X_background = X_train.toarray()
        X_eval = X_test.toarray()
    else:
        X_background = X_train
        X_eval = X_test

    explainer = shap.Explainer(model, X_background)
    shap_values = explainer(X_eval[:len(X_eval)])
    if isinstance(shap_values.values, list):
        shap_values_array = np.array(shap_values.values)
    else:
        shap_values_array = shap_values.values
    shap_values_dict[model_name] = shap_values
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = [(vectorizer.get_feature_names_out()[i], round(mean_abs_shap[i], 4)) for i in top_indices]
    top_shap_features[model_name] = top_features

    print(f"\n {model_name}")
    for feature, val in top_features:
        print(f"{feature:<20} → {val}")

# 7. Contextes de la feature SHAP 'not'
print("\nContextes pour la feature SHAP 'not':\n")
feature = "not"
matches = df[df["description"].str.contains(fr"\b{feature}\b", case=False, na=False)]
shown = 0
for idx, text in matches["description"].items():
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() == feature.lower() and i >= 5 and i + 6 <= len(words):
            context = " ".join(words[i-6:i+10])
            print(f"- {context}")
            shown += 1
            break
    if shown >= 5:
        break
print()

# 8. Statistiques sur le rouge

for color in ["red", "reddish"]:
    contains_color = df["description"].str.contains(fr"\b{color}\b", case=False, na=False)
    subset = df[contains_color]
    edible_count = subset["edibility_binary"].value_counts()

    print(f"\nStatistiques pour les descriptions contenant '{color}':")
    total = edible_count.sum()
    for label, count in edible_count.items():
        pct = (count / total) * 100
        print(f"- {label}: {count} ({pct:.1f}%)")