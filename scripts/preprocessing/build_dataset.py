import pandas as pd
import os

# Chemins
INPUT_PATH = "data/clean/champignons_with_description.csv"
OUTPUT_PATH = "data/features/mushrooms_dataset.csv"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Charger les données
df = pd.read_csv(INPUT_PATH)

# Garder uniquement les colonnes utiles
df = df[["description", "edibility_binary"]]

# Nettoyage de base
df = df.dropna(subset=["description", "edibility_binary"])
df["description"] = df["description"].str.strip()
df["edibility_binary"] = df["edibility_binary"].str.strip().str.lower()

# Sauvegarde
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset saved to {OUTPUT_PATH}")
print(f"Total examples: {len(df)}")
print(f"Class balance:\n{df['edibility_binary'].value_counts()}")
