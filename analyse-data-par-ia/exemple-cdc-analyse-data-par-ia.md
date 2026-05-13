# 📋 Cahier des Charges — Module Odoo "Data Analyst IA"

---

## 1. 🎯 Contexte et Origine du Projet

Ce projet est né d'un besoin concret d'analyse de données de production industrielle. Deux cas d'usage initiaux ont démontré la pertinence d'un outil générique :

> **Cas 1** — *"Le nombre de pièces fabriquées a-t-il une incidence sur le nombre de rebuts, article par article et globalement ?"*

> **Cas 2** — *"La couleur de l'OF précédent impacte-t-elle le taux de rebuts de l'OF en cours ? Passer de foncé à clair génère-t-il des tâches sur pièces plastiques ?"*

Ces deux exemples illustrent un besoin plus large : **permettre à n'importe quel utilisateur métier de poser une hypothèse en langage naturel et obtenir une réponse statistique fiable avec graphique**, sans compétences en data science ni en SQL.

---

## 2. 📌 Objectif du Module

Développer un module Odoo générique appelé **"Data Analyst IA"** permettant à tout utilisateur de :

1. Poser une hypothèse ou question en **langage naturel** (français ou anglais)
2. Obtenir automatiquement une **analyse statistique** appropriée
3. Visualiser les résultats sous forme de **graphiques**
4. Recevoir une **conclusion claire** (hypothèse confirmée ou infirmée, avec niveau de confiance)

Le tout **sans écrire une seule ligne de code**, sans accès direct à la base de données, et avec les données qui **restent sur le serveur**.

---

## 3. 👥 Utilisateurs Cibles

| Profil | Usage |
|---|---|
| Responsable production | Analyse qualité, rebuts, TRS |
| Responsable logistique | Retards, transporteurs, zones |
| Commercial | CA, marges, taux de conversion |
| RH | Absences, turnover, performance |
| Responsable qualité | Non-conformités, fournisseurs |
| Contrôleur de gestion | Coûts, marges, DSO |
| Administrateur Odoo | Configuration des sources de données |

---

## 4. 🏗️ Architecture Technique

### 4.1 Vue globale

```
┌──────────────────────────────────────────────────────────────┐
│                         ODOO                                  │
│                                                               │
│  Utilisateur          Module IA            Résultat           │
│  (prompt texte) ───▶  (pipeline) ───────▶ (graphique +       │
│                                            conclusion)        │
└──────────────────────────┬───────────────────────────────────┘
                           │ Schéma uniquement (pas de données)
                ┌──────────▼──────────┐
                │   LLM (IA locale    │
                │   via Ollama)       │
                │   ou Cloud API      │
                └──────────┬──────────┘
                           │ SQL généré
                ┌──────────▼──────────┐
                │  Export CSV         │
                │  temporaire         │
                │  (données isolées)  │
                └──────────┬──────────┘
                           │ CSV en lecture seule
                ┌──────────▼──────────┐
                │  Docker Sandbox     │
                │  Python isolé       │
                │  → PNG graphique    │
                └─────────────────────┘
```

### 4.2 Pipeline en 6 étapes

```
ÉTAPE 1 — Utilisateur saisit sa question en langage naturel
ÉTAPE 2 — L'IA génère un SQL SELECT (extraction des données)
ÉTAPE 3 — Odoo exécute le SQL (read-only, via ORM) → Export CSV temporaire
ÉTAPE 4 — L'IA génère un script Python d'analyse (ne connaît que le CSV)
ÉTAPE 5 — Python s'exécute dans un container Docker isolé → graphique PNG
ÉTAPE 6 — L'IA interprète les résultats → conclusion en français
           CSV temporaire supprimé immédiatement 🗑️
```

---

## 5. 🔒 Sécurité — Exigences Critiques

### 5.1 Protection de la base de données

| Règle | Détail |
|---|---|
| **Lecture seule** | Seuls les SELECT sont autorisés |
| **Whitelist des tables** | L'admin déclare explicitement les tables accessibles |
| **Colonnes sensibles bloquées** | password, token, secret… jamais exportés |
| **Limite de volume** | Maximum 100 000 lignes par export |
| **ORM Odoo** | Pas de connexion directe à PostgreSQL |

### 5.2 Sandbox Python (Docker)

| Règle | Détail |
|---|---|
| **Isolation totale** | Container Docker jetable par analyse |
| **Pas d'accès réseau** | `network_disabled = True` |
| **Pas d'accès fichiers** | Seul le CSV fourni est accessible en lecture |
| **Limites ressources** | Max 256 Mo RAM, 50% CPU, 2 min d'exécution |
| **Pas de droits root** | Exécution en user `nobody` |
| **Destruction auto** | Container supprimé après exécution |

### 5.3 Protection des données vers l'IA

| Règle | Détail |
|---|---|
| **Schéma uniquement** | Seule la structure des tables est envoyée au LLM |
| **Jamais de données brutes** | Les valeurs réelles ne quittent pas le serveur |
| **Ollama local privilégié** | 0 donnée qui sort de l'entreprise |

---

## 6. 🧩 Composants du Module

### 6.1 Modèles de données

```
ai.analysis.datasource    → Catalogue des tables autorisées (admin)
ai.analysis.column        → Colonnes disponibles par table
ai.analysis.template      → Bibliothèque de questions types
ai.analysis.query         → Les analyses (historique complet)
```

### 6.2 Services techniques

```
llm_service.py            → Appels API IA (Ollama ou Cloud)
csv_exporter.py           → Odoo → CSV sécurisé
docker_sandbox.py         → Exécution Python isolée
template_engine.py        → Remplissage des templates de code
```

---

## 7. 📊 Types d'Analyses Supportées

| Pattern | Déclencheur | Méthode statistique | Graphique |
|---|---|---|---|
| **Corrélation** | 2 variables numériques | Pearson + R² | Scatter + droite de régression |
| **Comparaison groupes** | 1 catégorie + 1 mesure | ANOVA + T-test | Boxplot |
| **Transitions séquences** | Ordre temporel + catégorie | LAG + ANOVA | Heatmap + Boxplot |
| **Série temporelle** | Variable + date | Régression temporelle | Lineplot + tendance |
| **Avant / Après** | 2 périodes comparées | T-test apparié | Barplot comparatif |
| **Distribution** | 1 variable seule | Stats descriptives | Histogramme |

---

## 8. ⚙️ Gestion Multi-Utilisateurs

### 8.1 File d'attente

- Maximum **N analyses en parallèle** (paramétrable par l'admin)
- File d'attente avec position visible par l'utilisateur
- Exécution en **arrière-plan** (ne bloque pas l'interface Odoo)
- Module OCA **Queue Job** recommandé

### 8.2 Notifications

- Notification temps réel quand l'analyse est terminée (`bus.bus`)
- Notification en cas d'erreur avec message explicite
- Indicateur de progression visible dans l'interface

---

## 9. 🤖 Intelligence Artificielle

### 9.1 Approche retenue : Templates + IA

```
L'IA ne génère pas le code from scratch.
Elle identifie le pattern et extrait les variables.
Le module remplit ensuite un template préécrit et testé.

→ Fiabilité maximale avec un modèle léger
```

### 9.2 Modèles compatibles

| Modèle | Type | RAM | Recommandation |
|---|---|---|---|
| **Mistral 7B** (Ollama) | Local | 8 Go | PME — usage courant |
| **CodeLlama 34B** (Ollama) | Local | 24 Go | PME — meilleure qualité |
| **Qwen2.5-Coder 32B** (Ollama) | Local | 20 Go | ⭐ Recommandé |
| **GPT-4o** (OpenAI) | Cloud | — | Fallback ou données non sensibles |
| **Mistral Large** (Mistral AI) | Cloud EU | — | Alternative RGPD |

### 9.3 Configuration dans Odoo

- URL du LLM configurable (`ir.config_parameter`)
- Modèle configurable
- Clé API configurable
- **Fallback automatique** Cloud si LLM local indisponible

---

## 10. 🖥️ Infrastructure Recommandée

### Petite entreprise (< 50 utilisateurs)

```
Serveur : 32 Go RAM + GPU 8-16 Go (RTX 3080/4080)
Modèle  : Mistral 7B ou Llama 3.1 8B via Ollama
Coût    : ~1 500 € matériel + 0 €/mois
```

### PME (50–200 utilisateurs)

```
Serveur : 64 Go RAM + GPU 24 Go (RTX 4090 ou A10)
Modèle  : Qwen2.5-Coder 32B ou CodeLlama 34B
Coût    : ~3 000 € matériel + 0 €/mois
```

### Sans GPU

```
Serveur : 16 Go RAM (serveur existant)
Modèle  : Mistral 7B quantisé Q4 via Ollama
Vitesse : ~30 sec/requête (acceptable)
Coût    : 0 € supplémentaire
```

---

## 11. 🖱️ Interface Utilisateur

### Écran principal

- Zone de **saisie libre** en langage naturel
- **Suggestions** de questions prédéfinies par domaine
- Indicateur de **statut** (En attente / En cours / Terminé)
- Affichage de la **conclusion** en langage naturel
- **1 ou 2 graphiques** selon le type d'analyse
- Affichage du **niveau de confiance** (%)

### Écran résultat

- Conclusion claire : *"Hypothèse CONFIRMÉE à 97.5%"*
- Statistiques détaillées (dépliables)
- Code SQL et Python généré (visible pour utilisateurs avancés)
- Boutons : **Export PDF** / **Partager** / **Sauvegarder**

### Écran administration

- Déclaration des tables et colonnes autorisées
- Description métier des tables (pour guider l'IA)
- Paramétrage du LLM (URL, modèle, clé API)
- Limite de parallélisme
- Historique de toutes les analyses

---

## 12. 🗂️ Cas d'Usage Métier Couverts

| Domaine | Exemples de questions |
|---|---|
| **Production** | Rebuts / couleur OF précédent, pannes / machine, TRS / équipe |
| **Qualité** | Non-conformités / fournisseur, coût qualité / période |
| **Logistique** | Retards / transporteur, retards / jour de semaine |
| **Commercial** | CA / commercial, panier moyen / mois, taux conversion |
| **RH** | Absences / équipe, turnover / département |
| **Finance** | Marges / famille produit, DSO / client |

---

## 13. 🚀 Roadmap de Développement

### Phase 1 — MVP (1 mois)

- [ ] Structure du module Odoo
- [ ] Catalogue des sources de données (admin)
- [ ] Interface utilisateur de base
- [ ] Intégration LLM (Ollama + OpenAI)
- [ ] Pipeline complet : prompt → SQL → CSV → Python → graphique
- [ ] 6 templates d'analyse précodés
- [ ] Docker sandbox
- [ ] Conclusion en langage naturel

### Phase 2 — Enrichissement (2 mois)

- [ ] File d'attente multi-utilisateurs (Queue Job)
- [ ] Notifications temps réel
- [ ] Historique et favoris des analyses
- [ ] Export PDF rapport complet
- [ ] Partage entre utilisateurs
- [ ] Bibliothèque de questions types par domaine

### Phase 3 — Avancé (3 mois)

- [ ] Détection automatique d'anomalies proactive
- [ ] Suggestions automatiques (*"Nous avons détecté une corrélation..."*)
- [ ] Dashboard des insights découverts
- [ ] Comparaisons automatiques période N vs N-1
- [ ] Mémoire des analyses pour suivi dans le temps

---

## 14. 📦 Dépendances Techniques

```
Odoo          : v16 ou v17
Python        : 3.10+
Docker        : 20.x+
Ollama        : dernière version stable

Librairies Python (dans le container Docker) :
  pandas, numpy, scipy, matplotlib, seaborn

Module Odoo OCA :
  queue_job (file d'attente)

API compatibles :
  OpenAI API format (OpenAI, Ollama, Mistral, LM Studio...)
```

---

## 15. ✅ Critères de Recette

| Critère | Condition de validation |
|---|---|
| Fiabilité analyse | 95% des prompts métier génèrent une analyse correcte |
| Sécurité BDD | Aucune écriture possible, tables non whitelistées inaccessibles |
| Isolation sandbox | Le code Python ne peut accéder à rien hors CSV |
| Performance | Résultat en moins de 2 minutes pour 50 000 lignes |
| Multi-utilisateurs | 5 analyses simultanées sans dégradation |
| Données locales | 0 donnée brute envoyée à une API externe |

---

> 💡 **Ce module représente une brique BI conversationnelle native dans Odoo**, comparable aux fonctionnalités premium de ThoughtSpot ou Power BI Copilot, mais 100% intégrée, 100% on-premise, et adaptée aux PME industrielles.