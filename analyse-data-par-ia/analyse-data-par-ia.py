#!/usr/bin/env python3
"""
Pipeline d'analyse statistique à partir d'un CSV.

Étape 1 (optionnelle) : export_csv()  — interroge la DB, produit data.csv
Étape 2              : analyser_csv() — lit uniquement le CSV, sans DB
Étape 3              : generer_html() — produit le rapport HTML

L'étape 2 est conçue pour tourner en sandbox Docker (pas de réseau, pas de DB).
"""

import base64
import csv as csv_module
import io
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # pas d'affichage graphique
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde

from config import (
    ANALYSE_GROUPE_COL,
    ANALYSE_X_COL,
    ANALYSE_Y_COL,
    ANALYSE_Y_MAX,
    POSTGRES_DB,
    POSTGRES_PASSWORD,
    POSTGRES_SQL,
    POSTGRES_USERNAME,
    PROMPT,
)

BASE_DIR   = Path(__file__).parent
OUTPUT_CSV  = BASE_DIR / "data.csv"
OUTPUT_HTML = BASE_DIR / "rapport.html"


# ---------------------------------------------------------------------------
# ÉTAPE 1 — Export DB → CSV  (ne tourne PAS dans la sandbox)
# ---------------------------------------------------------------------------

def export_csv() -> int:
    """Exécute POSTGRES_SQL et écrit le résultat dans OUTPUT_CSV.
    Retourne le nombre de lignes exportées."""
    import psycopg2  # import local : absent de la sandbox

    conn_params = {"dbname": POSTGRES_DB, "user": POSTGRES_USERNAME}
    if POSTGRES_PASSWORD:
        conn_params["password"] = POSTGRES_PASSWORD

    try:
        conn = psycopg2.connect(**conn_params)
    except psycopg2.OperationalError as e:
        print(f"Erreur de connexion : {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            cur.execute(POSTGRES_SQL)
            colonnes = [desc.name for desc in cur.description]
            lignes   = cur.fetchall()
    finally:
        conn.close()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(colonnes)
        writer.writerows(lignes)

    print(f"[export] {len(lignes)} ligne(s) → {OUTPUT_CSV}")
    return len(lignes)


# ---------------------------------------------------------------------------
# ÉTAPE 2 — Analyse CSV  (tourne aussi bien en local qu'en sandbox Docker)
# ---------------------------------------------------------------------------

def analyser_csv(
    csv_path: Path,
    question: str,
    x_col: str,
    y_col: str,
    groupe_col: Optional[str] = None,
    y_max: Optional[float] = None,
) -> dict:
    """
    Analyse la relation entre x_col et y_col dans le CSV.
    Retourne un dict avec :
      - stats     : dict de statistiques clés
      - conclusion: texte synthétique
      - graphique : image PNG encodée en base64
    
    Cette fonction est le point d'entrée que l'IA appellera avec ses paramètres.
    """
    df = pd.read_csv(csv_path)

    # Filtre outliers sur y
    if y_max is not None and y_col in df.columns:
        n_avant = len(df)
        df = df[pd.to_numeric(df[y_col], errors="coerce") <= y_max]
        n_exclus = n_avant - len(df)
        if n_exclus:
            print(f"[filtre] {n_exclus} ligne(s) exclue(s) : {y_col} > {y_max}")

    # Vérifications
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' absente du CSV. Colonnes disponibles : {list(df.columns)}")

    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    x_is_numeric = len(x) >= 2

    if x_is_numeric:
        df_clean = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    else:
        # x_col est catégoriel (texte) : on garde y numérique uniquement
        df_clean = df[[x_col, y_col]].copy()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[y_col])

    n = len(df_clean)
    y_vals = pd.to_numeric(df_clean[y_col], errors="coerce")
    x_vals = pd.to_numeric(df_clean[x_col], errors="coerce") if x_is_numeric else df_clean[x_col]

    # --- Statistiques ---
    if x_is_numeric and len(x_vals.dropna()) >= 2:
        r, p_value = stats.pearsonr(x_vals, y_vals)
        slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
    else:
        r, p_value, slope, intercept = 0.0, 1.0, 0.0, float(y_vals.mean()) if len(y_vals) else 0.0

    # --- Analyse par quartile de x (numérique uniquement) ---
    if x_is_numeric:
        q25 = x_vals.quantile(0.25)
        q75 = x_vals.quantile(0.75)
        mask_bas  = x_vals <= q25
        mask_haut = x_vals >= q75
        df_bas  = df_clean[mask_bas]
        df_haut = df_clean[mask_haut]
        r_bas,  p_bas  = stats.pearsonr(df_bas[x_col],  df_bas[y_col])  if len(df_bas)  > 2 else (0, 1)
        r_haut, p_haut = stats.pearsonr(df_haut[x_col], df_haut[y_col]) if len(df_haut) > 2 else (0, 1)
        y_moy_bas  = round(float(df_bas[y_col].mean()),  2) if len(df_bas)  else 0.0
        y_moy_haut = round(float(df_haut[y_col].mean()), 2) if len(df_haut) else 0.0
        effet_seuil = y_moy_bas > y_moy_haut * 2
    else:
        q25 = q75 = r_bas = r_haut = y_moy_bas = y_moy_haut = 0
        effet_seuil = False

    stats_dict = {
        "n_lignes"              : n,
        "correlation_r_global"  : round(float(r), 4),
        "p_value_global"        : round(float(p_value), 6),
        "pente_regression"      : round(float(slope), 4),
        "x_moyenne"             : round(float(x_vals.mean()), 2) if x_is_numeric else "—",
        "y_moyenne"             : round(float(y_vals.mean()), 2),
        "x_col"                 : x_col,
        "y_col"                 : y_col,
    }
    if x_is_numeric:
        stats_dict[f"y_moy (x ≤ Q25={q25:.0f})"] = y_moy_bas
        stats_dict[f"y_moy (x ≥ Q75={q75:.0f})"] = y_moy_haut
        stats_dict["r quartile bas"]  = round(float(r_bas), 4)
        stats_dict["r quartile haut"] = round(float(r_haut), 4)

    # --- Interprétation ---
    seuil_p = 0.05
    significatif = p_value < seuil_p
    forte = abs(r) >= 0.4
    force = (
        "très forte" if abs(r) >= 0.7
        else "modérée" if abs(r) >= 0.4
        else "faible"
    )
    direction = "positive" if r > 0 else "négative"
    confiance = round((1 - p_value) * 100, 1) if p_value < 1 else 0.0

    if significatif and forte:
        conclusion = (
            f"✅ Hypothèse CONFIRMÉE (p={p_value:.4f} < {seuil_p}, r={r:.3f}).\n"
            f"La corrélation entre « {x_col} » et « {y_col} » est {force} et {direction}. "
            f"Niveau de confiance statistique : {confiance:.1f} %.\n"
            f"Interprétation : quand {x_col} augmente de 1 unité, {y_col} varie de {slope:.4f} unité(s) en moyenne.\n"
            f"ℹ️  Seuils de référence : r ≥ 0.4 = corrélation modérée, r ≥ 0.7 = forte, r ≥ 0.9 = très forte."
        )
    elif significatif and effet_seuil:
        conclusion = (
            f"✅ Hypothèse CONFIRMÉE pour les petites quantités (p={p_value:.4f} < {seuil_p}).\n"
            f"La corrélation globale est {force} (r={r:.3f}), mais l'effet est concentré sur les faibles valeurs de « {x_col} » :\n"
            f"  • Quartile bas  (≤ {q25:.0f}) : {y_col} moyen = {y_moy_bas}  (r={r_bas:.3f})\n"
            f"  • Quartile haut (≥ {q75:.0f}) : {y_col} moyen = {y_moy_haut}  (r={r_haut:.3f})\n"
            f"Interprétation : quand « {x_col} » est faible, « {y_col} » est nettement plus élevé. "
            f"La relation est non linéaire (effet de petits lots).\n"
            f"ℹ️  Seuils de référence : r ≥ 0.4 = corrélation modérée, r ≥ 0.7 = forte, r ≥ 0.9 = très forte."
        )
    elif significatif and not forte:
        conclusion = (
            f"⚠️  Hypothèse PARTIELLEMENT CONFIRMÉE (p={p_value:.4f} < {seuil_p}, r={r:.3f}).\n"
            f"La relation existe statistiquement (non due au hasard) mais la corrélation est {force} — "
            f"elle n'est pas suffisamment prononcée pour être prédictive.\n"
            f"D'autres facteurs expliquent probablement mieux « {y_col} ».\n"
            f"ℹ️  Seuils de référence : r ≥ 0.4 = corrélation modérée, r ≥ 0.7 = forte, r ≥ 0.9 = très forte."
        )
    else:
        conclusion = (
            f"❌ Hypothèse NON CONFIRMÉE (p={p_value:.4f} ≥ {seuil_p}, r={r:.3f}).\n"
            f"Aucune relation statistiquement significative entre « {x_col} » et « {y_col} » "
            f"n'a été détectée sur cet échantillon de {n} observations.\n"
            f"ℹ️  Seuils de référence : r ≥ 0.4 = corrélation modérée, r ≥ 0.7 = forte, r ≥ 0.9 = très forte."
        )

    # --- Graphique interactif Plotly ---

    # Colonnes disponibles pour l'infobulle
    cols_info = [c for c in ["id", "heure_debut", "code_article", "moule", "presse_id"] if c in df.columns]
    customdata = list(zip(*[df[c].astype(str) for c in cols_info])) if cols_info else None
    hovertemplate = "<br>".join(f"<b>{c}</b> : %{{customdata[{i}]}}" for i, c in enumerate(cols_info))
    hovertemplate += f"<br><b>{x_col}</b> : %{{x}}<br><b>{y_col}</b> : %{{y:.2f}}<extra></extra>"

    # Densité locale pour colorier les points (noir=dense, gris clair=rare)
    try:
        y_num = pd.to_numeric(df_clean[y_col], errors="coerce")
        x_num = pd.to_numeric(df_clean[x_col], errors="coerce")
        x_is_cat = x_num.isna().all()

        if x_is_cat:
            # Densité sur Y uniquement, calculée par catégorie
            density = np.zeros(len(df_clean))
            for cat, grp_idx in df_clean.groupby(x_col).groups.items():
                y_grp = y_num.loc[grp_idx].dropna()
                if len(y_grp) >= 5:
                    kde_y = gaussian_kde(y_grp)
                    density[df_clean.index.get_indexer(y_grp.index)] = kde_y(y_grp)
        else:
            valid_kde = x_num.notna() & y_num.notna()
            if valid_kde.sum() < 10:
                raise ValueError("pas assez de points")
            xy = np.vstack([x_num[valid_kde], y_num[valid_kde]])
            kde = gaussian_kde(xy)
            density = kde(xy)

        d_min, d_max = density.min(), density.max()
        density_norm = (density - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(density)
        marker_color = density_norm
        colorscale = [[0, "rgb(210,210,210)"], [1, "rgb(10,10,10)"]]
        showscale = False
    except Exception as e:
        marker_color = "steelblue"
        colorscale = None
        showscale = False

    marker_cfg = dict(size=5, opacity=0.7)
    if colorscale:
        marker_cfg.update(color=marker_color, colorscale=colorscale, showscale=showscale)
    else:
        marker_cfg["color"] = marker_color

    scatter = go.Scatter(
        x=df_clean[x_col],
        y=df_clean[y_col],
        mode="markers",
        marker=marker_cfg,
        customdata=customdata if customdata else None,
        hovertemplate=hovertemplate,
        name="OF",
    )
    traces = [scatter]
    if x_is_numeric:
        x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
        regression = go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            line=dict(color="crimson", width=2),
            name=f"régression linéaire (r={r:.3f})",
            hoverinfo="skip",
        )
        traces.append(regression)

    fig_plotly = go.Figure(data=traces)
    fig_plotly.update_layout(
        title=dict(text=f"{question}<br><sup>r={r:.3f}  p={p_value:.4f}</sup>", font=dict(size=13)),
        xaxis=dict(title=x_col),
        yaxis=dict(title=y_col),
        hovermode="closest",
        height=480,
        margin=dict(t=80, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    graphique_html = fig_plotly.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    graphique_b64 = None  # plus utilisé

    # --- Stats descriptives par groupe (quand groupe_col == x_col) ---
    stats_par_groupe = []
    if groupe_col and groupe_col in df.columns and groupe_col == x_col:
        for g, grp in df.groupby(groupe_col):
            gy = pd.to_numeric(grp[y_col], errors="coerce").dropna()
            if len(gy) < 3:
                continue
            stats_par_groupe.append({
                "groupe" : g,
                "n"      : int(len(gy)),
                "moy"    : round(float(gy.mean()), 3),
                "mediane": round(float(gy.median()), 3),
                "std"    : round(float(gy.std()), 3),
                "max"    : round(float(gy.max()), 3),
            })

    # --- Corrélations par groupe ---
    corr_par_groupe = []
    exclus = []
    if groupe_col and groupe_col in df.columns and groupe_col != x_col:
        for g, grp in df.groupby(groupe_col):
            gx = pd.to_numeric(grp[x_col], errors="coerce")
            gy = pd.to_numeric(grp[y_col], errors="coerce")
            valid = gx.notna() & gy.notna()
            ng = valid.sum()
            if ng < 10:
                exclus.append({"groupe": str(g), "n": int(ng), "raison": f"moins de 10 observations (n={ng})"})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rg, pg = stats.pearsonr(gx[valid], gy[valid])
                rho_g, p_rho_g = stats.spearmanr(gx[valid], gy[valid])
            if np.isnan(rg) or np.isnan(rho_g):
                exclus.append({"groupe": str(g), "n": int(ng), "raison": "taux_rebut constant (variance nulle, r incalculable)"})
                continue
            # Filtrage sur Spearman (plus robuste pour relations non-linéaires)
            if p_rho_g >= 0.05:
                exclus.append({"groupe": str(g), "n": int(ng), "raison": f"corrélation non significative (ρ Spearman p={p_rho_g:.4f} ≥ 0.05)"})
                continue
            moy_x = gx[valid].mean()
            moy_y_pond = float(np.average(gy[valid], weights=gx[valid])) if gx[valid].sum() > 0 else float(gy[valid].mean())
            corr_par_groupe.append({
                "groupe"      : str(g),
                "n"           : int(ng),
                "r"           : round(float(rg), 4),
                "p_value"     : round(float(pg), 6),
                "rho"         : round(float(rho_g), 4),
                "p_rho"       : round(float(p_rho_g), 6),
                "significatif": True,
                # Force basée sur Spearman (insensible à la non-linéarité)
                "force"       : "très forte" if abs(rho_g) >= 0.7 else "modérée" if abs(rho_g) >= 0.4 else "faible",
                "moy_x"       : round(float(moy_x), 0),
                "moy_y_pond"  : round(moy_y_pond, 2),
            })
        corr_par_groupe.sort(key=lambda c: abs(c["rho"]), reverse=True)

    return {
        "question"         : question,
        "stats"            : stats_dict,
        "conclusion"       : conclusion,
        "graphique"        : graphique_html,
        "corr_par_groupe"  : corr_par_groupe,
        "stats_par_groupe" : stats_par_groupe,
        "exclus"           : exclus,
        "groupe_col"       : groupe_col,
        "x_col"            : x_col,
        "y_col"            : y_col,
    }


# ---------------------------------------------------------------------------
# ÉTAPE 3 — Génération du rapport HTML
# ---------------------------------------------------------------------------

def generer_html(resultat: dict, output_path: Path) -> None:
    """Génère un rapport HTML autonome (image embarquée en base64)."""
    stats = resultat["stats"]
    rows_html = "\n".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        for k, v in stats.items()
    )
    conclusion_html = resultat["conclusion"].replace("\n", "<br>")

    # Tableau stats descriptives par groupe (quand groupe_col == x_col)
    table_groupe_html = ""
    if resultat.get("stats_par_groupe"):
        groupe_col = resultat["groupe_col"]
        y_col = resultat["y_col"]
        lignes = ""
        for s in resultat["stats_par_groupe"]:
            g = s['groupe']
            # Formatage : si entier (heure), afficher "XXh", sinon texte brut
            try:
                g_fmt = f"{int(g):02d}h"
            except (ValueError, TypeError):
                g_fmt = str(g)
            lignes += (
                f"<tr>"
                f"<td style='text-align:center'>{g_fmt}</td>"
                f"<td style='text-align:right'>{s['n']}</td>"
                f"<td style='text-align:right'>{s['moy']:.3f} %</td>"
                f"<td style='text-align:right'>{s['mediane']:.3f} %</td>"
                f"<td style='text-align:right'>{s['std']:.3f}</td>"
                f"<td style='text-align:right'>{s['max']:.3f} %</td>"
                f"</tr>\n"
            )
        table_groupe_html = f"""
  <h2>Statistiques de {y_col} par {groupe_col}</h2>
  <table class="sortable">
    <thead><tr>
      <th>{groupe_col}</th>
      <th style='text-align:right'>n</th>
      <th style='text-align:right'>Moyenne</th>
      <th style='text-align:right'>Médiane</th>
      <th style='text-align:right'>Écart-type</th>
      <th style='text-align:right'>Max</th>
    </tr></thead>
    <tbody>{lignes}</tbody>
  </table>"""

    elif resultat.get("corr_par_groupe") and not resultat.get("stats_par_groupe"):
        groupe_col = resultat["groupe_col"]
        lignes = ""
        for c in resultat["corr_par_groupe"]:
            sig = "✅" if c["significatif"] else "—"
            force_color = (
                "#1a8a2e" if c["force"] == "très forte"
                else "#e67e22" if c["force"] == "modérée"
                else "#888"
            )
            moy_x_fmt = f"{c['moy_x']:,.0f}".replace(",", "\u202f")
            r_color = (
                "#1a8a2e" if abs(c['r']) >= 0.7
                else "#e67e22" if abs(c['r']) >= 0.4
                else "#888"
            )
            lignes += (
                f"<tr>"
                f"<td>{c['groupe']}</td>"
                f"<td style='text-align:right'>{moy_x_fmt}</td>"
                f"<td style='text-align:right'>{c['moy_y_pond']:.2f} %</td>"
                f"<td>{c['n']}</td>"
                f"<td style='color:{r_color}'>{c['r']:.2f}</td>"
                f"<td>{c['p_value']:.4f}</td>"
                f"<td style='color:{force_color};font-weight:bold'>{c['rho']:.2f}</td>"
                f"<td>{c['p_rho']:.4f}</td>"
                f"<td style='color:{force_color}'>{c['force']}</td>"
                f"</tr>\n"
            )
        table_groupe_html = f"""
  <h2>Corrélation par {groupe_col} (significatifs uniquement, triés par |ρ| décroissant)</h2>
  <p style='color:#555;font-size:0.9em'>ρ (Spearman) mesure les relations monotones même non-linéaires — plus adapté que r (Pearson) pour vos données.</p>
  <table class="sortable">
    <thead><tr>
      <th>{groupe_col}</th>
      <th style='text-align:right'>Moy. {resultat['x_col']}</th>
      <th style='text-align:right'>Taux rebut moyen pondéré</th>
      <th>n</th>
      <th>r (Pearson)</th><th>p-value</th>
      <th>ρ (Spearman)</th><th>p-value</th>
      <th>Force (ρ)</th>
    </tr></thead>
    <tbody>{lignes}</tbody>
  </table>"""

        # Tableau des exclus
        if resultat.get("exclus"):
            lignes_exclus = "\n".join(
                f"<tr><td>{e['groupe']}</td><td>{e['n']}</td><td>{e['raison']}</td></tr>"
                for e in sorted(resultat["exclus"], key=lambda e: e["groupe"])
            )
            table_groupe_html += f"""
  <h2 style="color:#888;font-size:1em;margin-top:30px">Articles exclus de l'analyse</h2>
  <table style="color:#888;font-size:0.9em">
    <thead><tr>
      <th>{groupe_col}</th><th>n</th><th>Raison d'exclusion</th>
    </tr></thead>
    <tbody>{lignes_exclus}</tbody>
  </table>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analyse statistique</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1   {{ font-size: 1.3em; color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
    .question {{ background: #eaf2ff; border-left: 4px solid #2980b9; padding: 12px 16px; border-radius: 4px; margin: 20px 0; font-style: italic; }}
    .conclusion {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 14px 16px; border-radius: 4px; margin: 24px 0; line-height: 1.7; }}
    .conclusion.rejected {{ background: #fdf2f2; border-color: #e74c3c; }}
    img  {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 20px 0; display: block; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th   {{ background: #1a5276; color: white; padding: 8px 12px; text-align: left; cursor: pointer; user-select: none; white-space: nowrap; }}
    th.sortable:hover {{ background: #21618c; }}
    th.sort-asc::after  {{ content: " ▲"; font-size: 0.75em; }}
    th.sort-desc::after {{ content: " ▼"; font-size: 0.75em; }}
    td   {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f7f9fc; }}
  </style>
  <script>
    document.addEventListener('DOMContentLoaded', function() {{
      document.querySelectorAll('table.sortable thead th').forEach(function(th, colIdx) {{
        th.classList.add('sortable');
        th.addEventListener('click', function() {{
          const table = th.closest('table');
          const tbody = table.querySelector('tbody');
          const rows  = Array.from(tbody.querySelectorAll('tr'));
          const asc   = th.classList.contains('sort-asc');
          // Réinitialiser toutes les flèches
          table.querySelectorAll('thead th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
          th.classList.add(asc ? 'sort-desc' : 'sort-asc');
          rows.sort(function(a, b) {{
            const va = a.cells[colIdx].getAttribute('data-val') ?? a.cells[colIdx].textContent.trim();
            const vb = b.cells[colIdx].getAttribute('data-val') ?? b.cells[colIdx].textContent.trim();
            const na = parseFloat(va.replace(/[^0-9.\-]/g, ''));
            const nb = parseFloat(vb.replace(/[^0-9.\-]/g, ''));
            const cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb : va.localeCompare(vb, 'fr', {{numeric: true}});
            return asc ? -cmp : cmp;
          }});
          rows.forEach(r => tbody.appendChild(r));
        }});
      }});
    }});
  </script>
</head>
<body>
  <h1>Rapport d'analyse statistique</h1>

  <div class="question">
    <strong>Question :</strong> {resultat['question']}
  </div>

  {resultat['graphique']}

  <div class="conclusion {'rejected' if '❌' in resultat['conclusion'] else ''}">
    <strong>Conclusion :</strong><br>{conclusion_html}
  </div>

  <h2>Statistiques détaillées</h2>
  <table>
    <thead><tr><th>Indicateur</th><th>Valeur</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  {table_groupe_html}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    print(f"[html] Rapport généré → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Étape 1 : export DB → CSV
    export_csv()

    # Étape 2 : analyse purement CSV (peut tourner en sandbox)
    resultat = analyser_csv(
        csv_path   = OUTPUT_CSV,
        question   = PROMPT,
        x_col      = ANALYSE_X_COL,
        y_col      = ANALYSE_Y_COL,
        groupe_col = ANALYSE_GROUPE_COL,
        y_max      = ANALYSE_Y_MAX,
    )

    print(f"\n{resultat['conclusion']}\n")

    # Étape 3 : rapport HTML
    generer_html(resultat, OUTPUT_HTML)
