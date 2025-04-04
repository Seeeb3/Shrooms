import pandas as pd
import json

# Charger le CSV binaire
df = pd.read_csv("data/clean/champignons_binary.csv")

# Construire le dictionnaire
mushroom_dict = dict(zip(df["name"], df["edibility_binary"]))

# Affichage
print("Exemple de dictionnaire (extrait) :")
for i, (name, label) in enumerate(mushroom_dict.items()):
    print(f"{name!r} : {label!r}")
    if i == 9:
        break

# Export en JSON
with open("data/dict/champignons_dict.json", "w", encoding="utf-8") as f:
    json.dump(mushroom_dict, f, indent=2, ensure_ascii=False)

print("\nDictionnaire exporté dans champignons_dict.json")
