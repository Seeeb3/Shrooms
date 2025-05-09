# 🍄 Mushroom Classification Project (Shrooms) 🍄

This project aims to collect, clean, and analyze data scraped from [rogersmushrooms.com](https://rogersmushrooms.com) in order to build a dataset of mushrooms annotated by edibility status (edible vs poisonous).

---

## 📌 Objectives

- Scrape mushroom pages from Rogers Mushrooms
- Extract scientific names, descriptions and edibility labels
- Clean and normalize the data
- Generate a labeled dataset: `edible` vs `poisonous`
- Retrieve mushroom descriptions from Wikipedia
- Build a dictionary of mushroom edibility
- Train and evaluate multiple ML classifiers
- Compare models based on precision, recall, F1-score
- Analyze feature importance using SHAP

---

## 🧪 Workflow / Pipeline

x
x
x

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```txt
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
imbalanced-learn
requests
beautifulsoup4
lxml
```

---

## ⚠️ Notes

- The current scraper retrieves **around ~300 mushrooms**, as many original pages are unavailable or broken on the Rogers Mushrooms site.
- Class imbalance is handled using `class_weight` and oversampling with SMOTE when applicable.
- All models are trained on the same fixed TF-IDF representation of the descriptions.

---

## ✍️ Authors

- [Sébastien DURNA](https://github.com/Seeeb3)
- [Manon MOULIN](https://github.com/Manon09825)
