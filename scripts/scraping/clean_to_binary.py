import pandas as pd

# Charger le fichier nettoyé
df = pd.read_csv("/data/clean/champignons_AZ_clean.csv")

# Supprimer les lignes où edibility_raw est NaN
df = df.dropna(subset=["edibility_raw"])

# Définir les classes binaires
edible_keywords = ["edible", "choice"]
poisonous_keywords = ["inedible", "poisonous/suspect", "deadly"]

# Création de la nouvelle colonne binaire
def to_binary_label(raw):
    raw = str(raw).strip().lower()
    if raw in edible_keywords:
        return "edible"
    elif raw in poisonous_keywords:
        return "poisonous"
    else:
        return None  # On peut filtrer ça aussi si besoin

# Appliquer la transformation
df["edibility_binary"] = df["edibility_raw"].apply(to_binary_label)

# Supprimer les lignes où on ne peut pas classer
df = df.dropna(subset=["edibility_binary"])

# Sauvegarde
df.to_csv("/data/clean/champignons_binary.csv", index=False)
print("Fichier nettoyé et binaire sauvegardé sous champignons_binary.csv")
