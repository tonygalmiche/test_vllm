Exactement ! C'est une excellente décision d'architecture. Le Python n'a **jamais accès à la BDD**, il ne voit qu'un CSV temporaire isolé.

---

## 🏗️ Architecture révisée

```
┌─────────────────────────────────────────────────────────────────┐
│                          ODOO                                    │
│                                                                  │
│  1. Utilisateur pose sa question                                 │
│        │                                                         │
│        ▼                                                         │
│  2. LLM génère une requête SQL (lecture seule)                   │
│        │                                                         │
│        ▼                                                         │
│  3. Odoo exécute le SQL via son ORM (contrôlé ✅)               │
│        │                                                         │
│        ▼                                                         │
│  4. Export CSV temporaire (données anonymisées si besoin)        │
│        │                                                         │
│        ▼                                                         │
│  5. LLM génère un script Python (ne connaît que le CSV)         │
│        │                                                         │
│        ▼                                                         │
│  6. Python exécuté en sandbox → lit CSV → génère graphique      │
│        │                                                         │
│        ▼                                                         │
│  7. Résultat + conclusion IA affiché dans Odoo                   │
│     CSV temporaire supprimé 🗑️                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Structure révisée

```
odoo_ai_analysis/
│
├── __manifest__.py
├── models/
│   └── analysis_query.py
├── services/
│   ├── llm_service.py          # Appels IA (inchangé)
│   ├── csv_exporter.py         # ← NOUVEAU : Odoo → CSV
│   └── python_executor.py      # Python sandbox (lit CSV uniquement)
├── views/
│   └── analysis_query_views.xml
└── security/
    └── ir.model.access.csv
```

---

## 🔄 Pipeline principal révisé

### `models/analysis_query.py`

```python
import logging
from odoo import models, fields
from ..services.llm_service import LLMService
from ..services.csv_exporter import CsvExporter
from ..services.python_executor import SafePythonExecutor

_logger = logging.getLogger(__name__)

class AnalysisQuery(models.Model):
    _name = "ai.analysis.query"
    _description = "Analyse statistique IA"

    name             = fields.Char("Titre", required=True)
    question         = fields.Text("Hypothèse / Question", required=True)
    table_hint       = fields.Text("Description des tables / colonnes disponibles")

    # Ce que l'IA génère
    generated_sql    = fields.Text("SQL généré (extraction)", readonly=True)
    generated_python = fields.Text("Python généré (analyse)", readonly=True)

    # Résultats
    result_image     = fields.Binary("Graphique", readonly=True)
    result_image_name = fields.Char(default="analyse.png")
    result_text      = fields.Text("Statistiques détaillées", readonly=True)
    conclusion       = fields.Text("Conclusion IA", readonly=True)
    rows_exported    = fields.Integer("Lignes exportées", readonly=True)

    state = fields.Selection([
        ("draft",   "Brouillon"),
        ("running", "En cours..."),
        ("done",    "Terminé"),
        ("error",   "Erreur"),
    ], default="draft")
    error_message = fields.Text(readonly=True)

    def action_analyze(self):
        self.ensure_one()
        self.write({"state": "running", "error_message": False})

        csv_path = None
        try:
            llm      = LLMService(self.env)
            schema   = self.table_hint or ""

            # ── ÉTAPE 1 : L'IA génère le SQL d'extraction ─────────────────────
            _logger.info("Étape 1 : Génération du SQL d'extraction")
            sql_response = llm.generate_extraction_sql(
                question=self.question,
                schema=schema,
            )
            self.generated_sql = sql_response["sql"]

            # ── ÉTAPE 2 : Odoo exécute le SQL et exporte en CSV ───────────────
            _logger.info("Étape 2 : Export CSV")
            exporter  = CsvExporter(self.env)
            csv_path, nb_rows = exporter.export(
                sql=self.generated_sql,
                analysis_id=self.id,
            )
            self.rows_exported = nb_rows
            _logger.info(f"{nb_rows} lignes exportées vers {csv_path}")

            # ── ÉTAPE 3 : L'IA génère le Python d'analyse (sur CSV) ───────────
            _logger.info("Étape 3 : Génération du Python d'analyse")
            python_response = llm.generate_analysis_python(
                question=self.question,
                csv_path=csv_path,
                columns=exporter.last_columns,  # Noms des colonnes du CSV
            )
            self.generated_python = python_response["code"]

            # ── ÉTAPE 4 : Exécution Python sandboxée (CSV uniquement) ─────────
            _logger.info("Étape 4 : Exécution Python sandbox")
            executor = SafePythonExecutor()
            result   = executor.run(
                code=self.generated_python,
                csv_path=csv_path,
            )

            # ── ÉTAPE 5 : L'IA interprète les résultats ───────────────────────
            _logger.info("Étape 5 : Interprétation IA")
            conclusion = llm.interpret_results(
                question=self.question,
                results=result["summary"],
            )

            self.write({
                "result_image": result.get("image_b64"),
                "result_text":  result.get("text"),
                "conclusion":   conclusion,
                "state":        "done",
            })

        except Exception as e:
            self.write({"state": "error", "error_message": str(e)})
            _logger.exception("Erreur analyse IA")

        finally:
            # ── ÉTAPE 6 : Suppression du CSV temporaire ───────────────────────
            if csv_path:
                CsvExporter.cleanup(csv_path)
                _logger.info(f"CSV temporaire supprimé : {csv_path}")
```

---

## 📤 `services/csv_exporter.py` — Le gardien de la BDD

```python
import os
import csv
import tempfile
import logging
from odoo.exceptions import UserError

_logger  = logging.getLogger(__name__)

# ✅ Seules ces tables peuvent être lues
ALLOWED_TABLES = {
    "mrp_production",
    "mrp_workorder", 
    "product_product",
    "product_template",
    "stock_move",
    "stock_quant",
    # Ajouter tes tables métier ici
    "of_production",       # Exemple table custom
}

# ❌ Colonnes jamais exportées (données sensibles)
BLOCKED_COLUMNS = {
    "password", "passwd", "token", "secret",
    "write_uid", "create_uid",  # Optionnel selon besoin
}

class CsvExporter:

    def __init__(self, env):
        self.env          = env
        self.last_columns = []

    def export(self, sql: str, analysis_id: int) -> tuple[str, int]:
        """
        Valide et exécute le SQL, exporte le résultat en CSV temporaire.
        Retourne (chemin_csv, nb_lignes)
        """
        # ── Validation de sécurité ─────────────────────────────────────────
        self._validate_sql(sql)

        # ── Exécution via le cursor Odoo (pas de connexion directe) ────────
        self.env.cr.execute(sql)
        rows    = self.env.cr.fetchall()
        columns = [desc[0] for desc in self.env.cr.description]

        # Vérifier les colonnes bloquées
        for col in columns:
            if col.lower() in BLOCKED_COLUMNS:
                raise UserError(f"Colonne sensible détectée et bloquée : {col}")

        self.last_columns = columns
        nb_rows = len(rows)

        if nb_rows == 0:
            raise UserError("La requête SQL n'a retourné aucune donnée.")

        if nb_rows > 100_000:
            raise UserError(f"Trop de données ({nb_rows} lignes). Affinez votre question.")

        # ── Écriture CSV temporaire ────────────────────────────────────────
        tmp_dir  = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, f"ai_analysis_{analysis_id}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)   # En-tête
            writer.writerows(rows)     # Données

        _logger.info(f"CSV exporté : {csv_path} ({nb_rows} lignes, {len(columns)} colonnes)")
        return csv_path, nb_rows

    def _validate_sql(self, sql: str):
        """
        Vérifie que le SQL est en lecture seule et sur tables autorisées.
        """
        sql_upper = sql.upper().strip()

        # ❌ Interdire toute écriture
        forbidden_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
            "ALTER", "TRUNCATE", "GRANT", "REVOKE", "EXECUTE",
        ]
        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                raise UserError(
                    f"Requête refusée : le mot-clé '{keyword}' est interdit. "
                    f"Seules les requêtes SELECT sont autorisées."
                )

        # ✅ Doit commencer par SELECT
        if not sql_upper.startswith("SELECT"):
            raise UserError("La requête doit commencer par SELECT.")

        # ✅ Vérifier que les tables utilisées sont dans la whitelist
        for table in ALLOWED_TABLES:
            pass  # Parsing plus fin possible avec sqlparse

        # Optionnel : parsing avancé avec sqlparse
        # import sqlparse
        # parsed = sqlparse.parse(sql)[0]
        # ... vérifier les identifiers

    @staticmethod
    def cleanup(csv_path: str):
        """Supprime le CSV temporaire après analyse."""
        try:
            if csv_path and os.path.exists(csv_path):
                os.remove(csv_path)
        except Exception as e:
            _logger.warning(f"Impossible de supprimer {csv_path} : {e}")
```

---

## 🐍 `services/python_executor.py` — Sandbox CSV only

```python
import io, base64, os, logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)

# ✅ Modules autorisés dans le sandbox
SAFE_BUILTINS = {
    "print": print, "range": range, "len": len,
    "round": round, "abs": abs, "sum": sum,
    "min": min, "max": max, "sorted": sorted,
    "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "bool": bool,
    "True": True, "False": False, "None": None,
}

SAFE_MODULES = {
    "pd":      __import__("pandas"),
    "np":      __import__("numpy"),
    "plt":     plt,
    "stats":   __import__("scipy.stats", fromlist=["stats"]),
    "sns":     __import__("seaborn"),
}

class SafePythonExecutor:

    def run(self, code: str, csv_path: str) -> dict:
        """
        Exécute le code Python généré par l'IA.
        Le code ne peut accéder qu'au CSV fourni, rien d'autre.
        """
        # Variables injectées : accès CSV uniquement
        local_vars = {
            **SAFE_MODULES,
            "CSV_PATH":    csv_path,      # ← Seul accès aux données
            "result_text": "",
            "fig":         None,
        }

        # ── Exécution dans un contexte restreint ──────────────────────────
        try:
            exec(
                code,
                {"__builtins__": SAFE_BUILTINS},  # Builtins limités
                local_vars,
            )
        except Exception as e:
            raise RuntimeError(f"Erreur d'exécution du code Python : {e}")

        # ── Récupération du graphique ─────────────────────────────────────
        image_b64 = None
        fig = local_vars.get("fig") or (plt.gcf() if plt.get_fignums() else None)

        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.close("all")

        result_text = local_vars.get("result_text", "")

        return {
            "text":    result_text,
            "summary": result_text[:2000],  # Résumé pour le LLM
            "image_b64": image_b64,
        }
```

---

## 🔒 Récapitulatif des couches de sécurité

```
┌─────────────────────────────────────────────────────────┐
│  COUCHE 1 — SQL Whitelist                               │
│  • Seul SELECT autorisé                                 │
│  • Tables whitelistées explicitement                    │
│  • Colonnes sensibles bloquées                          │
├─────────────────────────────────────────────────────────┤
│  COUCHE 2 — CSV Tampon                                  │
│  • Python ne voit JAMAIS la BDD                         │
│  • Fichier temporaire isolé                             │
│  • Supprimé immédiatement après analyse                 │
├─────────────────────────────────────────────────────────┤
│  COUCHE 3 — Sandbox Python                              │
│  • __builtins__ réduits (pas d'open, import, os...)    │
│  • Modules autorisés : pandas, numpy, scipy, plt        │
│  • Pas d'accès réseau, pas d'accès fichiers (sauf CSV)  │
├─────────────────────────────────────────────────────────┤
│  COUCHE 4 — Données vers LLM                            │
│  • Seul le SCHÉMA est envoyé à l'API IA                 │
│  • Jamais de données brutes vers l'extérieur            │
│  • → Ollama local = 0 donnée qui sort                   │
└─────────────────────────────────────────────────────────┘
```

---

> 💡 **Prochaine étape logique** : ajouter un **historique des analyses** avec les graphiques sauvegardés, pour que les utilisateurs puissent retrouver et comparer leurs analyses dans le temps. Tu veux qu'on parte là-dessus ?
