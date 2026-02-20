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

### Choix du fournisseur LLM

- `LLM_PROVIDER` : `"vllm"` (défaut) pour utiliser le serveur vLLM local, ou `"gemini"` pour utiliser Google Gemini

### Configuration vLLM

- `BASE_URL` : URL du serveur vLLM (ex. `http://localhost:8000/v1`)
- `MODEL_NAME` : nom du modèle exposé
- `API_KEY` : clé si nécessaire (ou `None` sinon)

### Configuration Google Gemini

- `GEMINI_API_KEY` : clé API Google Gemini (obligatoire si `LLM_PROVIDER = "gemini"`)
- `GEMINI_MODEL` : modèle Gemini à utiliser (défaut : `gemini-2.5-flash`)

Pour obtenir une clé API Gemini :

1. Rendez-vous sur **https://aistudio.google.com/apikey**
2. Connectez-vous avec votre compte Google
3. Cliquez sur **"Create API Key"**
4. Choisissez un projet Google Cloud existant ou laissez-en créer un nouveau
5. Copiez la clé générée (commence par `AIza...`)


### Limite de la version Gratuite de Gemini au 20/02/2026

Côté API, plusieurs retours indiquent qu’historiquement 2.5 Flash tournait autour de 250 requêtes/jour pour le palier gratuit, mais qu’en 2025–2026 beaucoup de comptes sont tombés à environ 20 requêtes/jour, sans plafond explicite par heure (on parle plutôt de « RPD » = requêtes par jour et éventuellement de limites par minute comme 10 RPM).

Comme ces limites sont à la fois floues, par produit, et régulièrement ajustées, le plus fiable pour ton cas d’usage précis (appli Gemini, web, API, etc.) reste de vérifier directement la page d’aide de ton compte Gemini ou la console de quotas associée, qui indiquent les chiffres effectifs appliqués à ton compte à un instant T.




### Paramètres communs

- `FACTURES_DIR` : dossier contenant les PDF à analyser
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





