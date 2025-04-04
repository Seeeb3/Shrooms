import pandas as pd

# Charger le CSV original
df = pd.read_csv("data/raw/champignons_AZ.csv")

# Supprimer les lignes où le nom est "Unknown"
df_clean = df[df["name"].str.lower() != "unknown"]

# Sauvegarder dans un nouveau fichier
df_clean.to_csv("data/clean/champignons_AZ_clean.csv", index=False)

print(f"Nettoyage terminé : {len(df) - len(df_clean)} lignes supprimées.")
print(f"Fichier sauvegardé sous : champignons_AZ_clean.csv")
