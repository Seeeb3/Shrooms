import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# Charger le CSV existant
df = pd.read_csv("champignons_binary.csv")

# Résultats
descriptions = []
not_found = []

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_description_from_page(url):
    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        # Chercher une balise <b>Description</b> suivie d'un <p>
        label = soup.find("b", string=lambda s: s and "description" in s.lower())
        if label:
            desc_p = label.find_next("p")
            if desc_p:
                return desc_p.get_text(strip=True)

        # Fallback : parfois le <p> descriptif est juste le premier <p> sous le tableau des images
        first_p = soup.find("p")
        if first_p:
            return first_p.get_text(strip=True)

    except Exception as e:
        print(f"Erreur pour {url} : {e}")
    return None

# Traitement ligne par ligne
for i, row in df.iterrows():
    name = row["name"]
    url = row["url"]
    print(f"[{i+1}/{len(df)}] {name}")

    description = get_description_from_page(url)
    if description:
        descriptions.append({
            "name": name,
            "edibility_binary": row["edibility_binary"],
            "url": url,
            "description": description
        })
    else:
        not_found.append(name)

    time.sleep(0.5)  # pour ne pas spammer le serveur

# Export CSV des descriptions trouvées
df_out = pd.DataFrame(descriptions)
df_out.to_csv("champignons_with_description.csv", index=False)
print("\nDescriptions sauvegardées dans champignons_with_description.csv")

# Affichage des non trouvés
if not_found:
    print("\nAucun paragraphe 'Description' trouvé pour :")
    for name in not_found:
        print(f"- {name}")
else:
    print("\nToutes les descriptions ont été trouvées !")
