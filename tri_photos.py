#!/usr/bin/env python3
import io
import time
import requests
import base64
import concurrent.futures
from pathlib import Path
from PIL import Image

from config import BASE_URL, DOSSIER_PHOTOS_A_TRIER, PHOTOS_BATCH_SIZE, MODEL_NAME, PHOTOS_MAX_RESOLUTION


PERSONNES = {
    "Tony": "Homme de 55 ans, cheveux courts chatain clair avec des lunettes et souvent avec un sourire",
    "Virginie": "Femme de 55 ans, cheveux foncés longs souvent avec des lunettes et un grand sourire et des fois mal rasé",
    "Jimmy": "Homme de 33 ans, cheveux courts chatain clair avec des lunettes, une barbe courte et une petite moustache. Aime faire la grimace ou défomer ses photos",
    "Natacha": "Femme de 30 ans, cheveux chatain foncé et souvent avec un sourire. Peux avoir des lunettes de soleil noire",


    "Inconnu": "Autre personne"
}

DGX_URL = f"{BASE_URL}/chat/completions"
BATCH_SIZE = PHOTOS_BATCH_SIZE  # Nombre de requêtes parallèles


def resize_photo(photo_path, max_resolution):
    """Redimensionne une photo en mémoire (sans modifier l'original).
    Retourne les données JPEG encodées en base64."""
    img = Image.open(photo_path)
    if max_resolution and max(img.size) > max_resolution:
        img.thumbnail((max_resolution, max_resolution), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def analyze_photo(photo_path):
    """Envoie UNE photo vers vLLM et renvoie (nom_fichier, label)."""
    img_data = resize_photo(photo_path, PHOTOS_MAX_RESOLUTION)

    liste = "\n".join(f"{i}. {nom}: {desc}" for i, (nom, desc) in enumerate(PERSONNES.items(), 1))
    noms = ", ".join(PERSONNES.keys())
    prompt = f"""CHOISIS EXACTEMENT UN NOM parmi :
{liste}

Réponds UNIQUEMENT: {noms}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }

    resp = requests.post(DGX_URL, json=payload, timeout=120)
    data = resp.json()

    if "choices" not in data:
        print(f"  ⚠ Erreur pour {photo_path.name}: {data}")
        return photo_path.name, "Inconnu"

    label = data["choices"][0]["message"]["content"].strip()
    for nom in PERSONNES:
        if nom.lower() in label.lower():
            return photo_path.name, nom
    return photo_path.name, "Inconnu"


def analyze_batch(photos):
    """Envoie un batch de photos en parallèle via ThreadPoolExecutor."""
    mapping = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(photos)) as executor:
        futures = {executor.submit(analyze_photo, p): p for p in photos}
        for future in concurrent.futures.as_completed(futures):
            name, label = future.result()
            mapping[name] = label
    return mapping


# Tri par batchs
dossier_photos = Path(DOSSIER_PHOTOS_A_TRIER)
dossier_sortie = Path("%s/photos_triees"%DOSSIER_PHOTOS_A_TRIER)
dossier_sortie.mkdir(exist_ok=True)

photos = list(dossier_photos.glob("*.jpg"))
stats = {nom: 0 for nom in PERSONNES}
nb_total = len(photos)
t_total_start = time.time()

print(f"{nb_total} photos à traiter (résolution max envoi : {PHOTOS_MAX_RESOLUTION or 'originale'})\n")

for i in range(0, nb_total, BATCH_SIZE):
    batch = photos[i:i+BATCH_SIZE]
    num_batch = i // BATCH_SIZE + 1
    print(f"Traitement batch {num_batch} ({len(batch)} photos)...")

    t_batch_start = time.time()
    labels = analyze_batch(batch)
    t_batch = time.time() - t_batch_start

    for photo_path, label in labels.items():
        src = dossier_photos / photo_path
        dest = dossier_sortie / label
        dest.mkdir(exist_ok=True)
        src.rename(dest / photo_path)
        stats[label] += 1
        print(f"  ✓ {photo_path} → {label}")

    print(f"  ⏱ Batch {num_batch} : {t_batch:.2f}s ({t_batch/len(batch):.2f}s/photo)\n")

t_total = time.time() - t_total_start
print(f"🎉 Tri terminé ! {stats}")
print(f"⏱ Durée totale : {t_total:.2f}s | Photos : {nb_total} | Moyenne : {t_total/max(nb_total,1):.2f}s/photo")
