# Test vLLM - Extraction de Factures

## Installation locale

```bash
cd /home/tony/Documents/Développement/dev_odoo/16.0/plastigray/test_vllm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Modifiez [config.py](config.py) pour définir :
- `FACTURES_DIR` : dossier contenant les PDF à analyser
- `BASE_URL` : URL du serveur vLLM (ex. `http://localhost:8000/v1`)
- `MODEL_NAME` : nom du modèle exposé
- `API_KEY` : clé si nécessaire (ou `None` sinon)
- `TEMPERATURE` : valeur de température transmise au modèle
- `CLIENT_HINTS` : liste indicative des clients attendus (sert de contexte dans le prompt et pour un rattrapage automatique si le modèle ne renvoie pas de nom)

## Exécution

```bash
python test_vllm.py
```

Le script écrit un JSON détaillant chaque facture sur la sortie standard puis affiche
sur la sortie d'erreur un tableau récapitulatif avec les montants, le client,
son identifiant (SIRET/SIREN), la durée par fichier, un indicateur de validation
(✅/❌) et la durée totale du traitement (avec moyenne et nombre de factures).
Chaque colonne affiche "❌" lorsqu'une information est manquante. En fin de tableau,
le script affiche aussi le nombre total de croix rouges, le nombre de factures
contenant au moins une anomalie et le pourcentage correspondant.
Une extraction est considérée valide uniquement si toutes les informations principales
(dont l'identifiant client) sont présentes, si le champ client n'est pas vide (ni
remplacé par "Inconnu") et si HT + TVA = TTC à ±0,05 € près.

Si le modèle laisse le champ client ou l'identifiant vide, le script envoie une
seconde requête plus ciblée pour extraire uniquement ces deux informations puis,
si besoin, tente un rattrapage automatique pour le nom du destinataire en
recherchant les valeurs de `CLIENT_HINTS` directement dans le texte extrait du
PDF. Les valeurs ainsi déduites sont affichées dans la sortie standard et
utilisées pour le tableau.





