import requests
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
import string
import os

BASE = "https://rogersmushrooms.com"
GALLERY_BASE = BASE + "/gallery/"

COMESTIBILITY_KEYWORDS = {
    "choice": "edible",
    "edible": "edible",
    "edible – good": "edible",
    "poisonous": "poisonous",
    "poisonous/suspect": "poisonous",
    "not edible": "inedible",
    "inedible": "inedible",
    "not recommended": "inedible",
    "toxic": "poisonous",
    "unknown": "unknown"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def estimate_edibility(text):
    text = text.lower().replace("\xa0", " ")
    for keyword, label in COMESTIBILITY_KEYWORDS.items():
        if keyword in text:
            return label
    return "unknown"

def get_edibility_raw(soup):
    try:
        edibility_row = soup.find("b", string=lambda s: s and "edibility:" in s.lower())
        if edibility_row:
            parent_td = edibility_row.find_parent("td")
            if parent_td:
                return parent_td.get_text(strip=True).replace("edibility:", "").strip()
    except Exception as e:
        print("Erreur extraction edibility:", e)
    return ""

def scrape_mushroom_page(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    h2 = soup.find("h2")
    name_match = re.match(r"([A-Za-z]+ [a-z]+)\.", h2.get_text(strip=True)) if h2 else None
    name = name_match.group(1) if name_match else "Unknown"

    edibility_text = get_edibility_raw(soup)

    return {
        "name": name,
        "edibility_raw": edibility_text,
        "url": url
    }

def get_page_url(letter, page_num):
    if page_num == 1:
        if letter == "a":
            return f"{GALLERY_BASE}default_GID_253_chr_{letter}.html"
        else:
            return f"{GALLERY_BASE}default_gid__chr_{letter}.html"
    else:
        return f"{GALLERY_BASE}default_gid__page_{page_num}_startPage_1_chr_{letter}.html"

def extract_links_from_gallery(letter):
    links = []
    page_num = 1

    while True:
        url = get_page_url(letter, page_num)
        print(f"Chargement page {page_num} pour lettre {letter.upper()}...")

        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code != 200:
                print(f"Fin pagination (status {res.status_code})")
                break

            soup = BeautifulSoup(res.text, "html.parser")
            page_links = []

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "DisplayBlock_bid_" in href:
                    full_url = BASE + "/gallery/" + href.split("/gallery/")[-1]
                    if full_url not in page_links:
                        page_links.append(full_url)

            if not page_links:
                print("Aucun lien trouvé, fin de la pagination.")
                break

            links.extend(page_links)
            page_num += 1
            time.sleep(1)
        except Exception as e:
            print(f"Erreur pendant la récupération de {url}: {e}")
            break

    return links

# ▶️ Scraping de toutes les lettres
results = []

for letter in string.ascii_lowercase:
    print(f"\nTraitement de la lettre {letter.upper()}")
    links = extract_links_from_gallery(letter)
    print(f"{len(links)} champignons trouvés pour {letter.upper()}")

    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Scraping {link}")
        try:
            data = scrape_mushroom_page(link)
            results.append(data)
            time.sleep(0.5)
        except Exception as e:
            print(f"Erreur scraping {link}: {e}")

# ▶️ Enregistrement final
output_path = "data/raw/champignons_AZ.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"\nScraping terminé. Résultats sauvegardés dans {output_path}")
