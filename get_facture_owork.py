#!/usr/bin/env python3
"""Extraction des dernières factures fournisseur via XML-RPC
et téléchargement des PDF depuis O'Work."""

from typing import List, Dict, Any, Optional, Sequence, Tuple
import http.cookiejar
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from config import (
    ODOO_DB,
    ODOO_PASSWORD,
    OWORK_BASE_URL,
    OWORK_INVOICE_DATE_FROM,
    OWORK_INVOICE_LIMIT,
    OWORK_PASSWORD,
    OWORK_REALM,
    OWORK_USERNAME,
)

from analyse_facture_common import odoo_connect

OUTPUT_DIR = Path(__file__).parent / "factures-owork"


# ---------------------------------------------------------------------------
# Odoo XML-RPC
# ---------------------------------------------------------------------------

def _sanitize(value: Any) -> str:
    """Normalise l'affichage pour le tableau textuel."""

    if value is None or value is False:
        return ""
    return str(value)


def _format_table(rows: Sequence[Tuple[str, ...]],
                  headers: Tuple[str, ...]) -> str:
    """Construit un tableau aligné en texte brut."""

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _join(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_join(headers), separator]
    lines.extend(_join(row) for row in rows)
    return "\n".join(lines)


def fetch_last_vendor_bills(limit: int = OWORK_INVOICE_LIMIT) -> List[Dict[str, Any]]:
    """Retourne les *limit* dernières factures fournisseur avec is_id_owork."""

    uid, models = odoo_connect()

    domain = [
        ("move_type", "=", "in_invoice"),
        ("state", "!=", "cancel"),
    ]
    if OWORK_INVOICE_DATE_FROM:
        domain.append(("invoice_date", ">=", OWORK_INVOICE_DATE_FROM))

    return models.execute_kw(
        ODOO_DB,
        uid,
        ODOO_PASSWORD,
        "account.move",
        "search_read",
        [domain],
        {
            "fields": ["id", "name", "invoice_date", "is_id_owork"],
            "limit": limit,
            "order": "invoice_date asc, id asc",
        },
    )


# ---------------------------------------------------------------------------
# O'Work – curl helpers
# ---------------------------------------------------------------------------

def _ensure_curl_present() -> None:
    if shutil.which("curl") is None:
        raise SystemExit("curl est introuvable dans le PATH")


def _curl_json(cmd: list) -> dict:
    """Exécute une commande curl et décode la réponse JSON."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl a échoué (code {result.returncode})\n{result.stderr}")
    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError("Réponse vide reçue depuis curl")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Impossible de décoder le JSON.\nDébut de la réponse:\n{raw[:500]}"
        )


def _extract_cookie(cookie_path: Path, name: str) -> Optional[str]:
    jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except FileNotFoundError:
        return None
    for cookie in jar:
        if cookie.name == name:
            return cookie.value
    return None


def owork_login(base_url: str, username: str, password: str,
                realm: str, cookie_path: Path) -> dict:
    """POST /api/auth/user → renvoie le JSON utilisateur."""
    url = base_url.rstrip("/") + "/api/auth/user"
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", "Content-Type: application/x-www-form-urlencoded",
        "--data-urlencode", f"username={username}",
        "--data-urlencode", f"password={password}",
        "--data-urlencode", f"realm={realm}",
        "-c", str(cookie_path),
    ]
    return _curl_json(cmd)


def owork_get_attachment(base_url: str, cookie_path: Path,
                         doc_id: int) -> Optional[dict]:
    """Récupère la pièce jointe (ir_attachment) du document O'Work *doc_id*."""
    url = base_url.rstrip("/") + "/api/db/multiread"
    payload = json.dumps({
        "reads": {
            "attach": {
                "table": "ir_attachment",
                "search": {"res_id": doc_id, "res_model": "iz.doc"},
            },
        }
    })
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-b", str(cookie_path),
        "-c", str(cookie_path),
        "-H", "Content-Type: application/json",
        "-d", payload,
    ]
    data = _curl_json(cmd)
    attach_list = data.get("attach", [])
    if not attach_list:
        return None
    return attach_list[0] if isinstance(attach_list, list) else attach_list


def owork_download_pdf(base_url: str, cookie_path: Path,
                       attach_id: int, filename: str,
                       output_path: Path) -> None:
    """GET /api/readBinary/download/<attach_id>/<filename> → fichier local."""
    encoded_name = urllib.parse.quote(filename, safe="")
    download_token = f"{attach_id}_{int(time.time() * 1000)}"
    url = (
        f"{base_url.rstrip('/')}/api/readBinary/download/"
        f"{attach_id}/{encoded_name}?downloadToken={download_token}"
    )
    cmd = [
        "curl", "-sS", "-L",
        "-b", str(cookie_path),
        "-o", str(output_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl download échoué (code {result.returncode})\n{result.stderr}")
    if output_path.stat().st_size == 0:
        raise RuntimeError("Le fichier téléchargé est vide")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_curl_present()
    t_start = time.time()

    # --- 1. Factures Odoo ---
    print("Récupération des dernières factures fournisseur depuis Odoo ...")
    try:
        bills = fetch_last_vendor_bills()
    except Exception as exc:
        print(f"Erreur Odoo: {exc}", file=sys.stderr)
        return

    if not bills:
        print("Aucune facture fournisseur trouvée.")
        return

    # Filtrer les factures ayant un is_id_owork
    bills_with_owork = [
        b for b in bills
        if b.get("is_id_owork") and str(b["is_id_owork"]).strip()
    ]
    if not bills_with_owork:
        print("\nAucune facture n'a de is_id_owork → rien à télécharger.")
        return

    print(f"\n{len(bills_with_owork)} facture(s) avec is_id_owork à télécharger.")

    # --- 2. Connexion O'Work ---
    base_url = OWORK_BASE_URL or "https://plastigray.oflux-owork.fr"
    print(f"\nConnexion à O'Work ({base_url}) ...")

    with tempfile.NamedTemporaryFile(prefix="owork_cookies_", delete=False) as tmp:
        cookie_path = Path(tmp.name)

    try:
        user = owork_login(base_url, OWORK_USERNAME, OWORK_PASSWORD,
                           OWORK_REALM, cookie_path)
        connect_sid = _extract_cookie(cookie_path, "connect.sid")
        if not connect_sid:
            raise RuntimeError("Cookie connect.sid introuvable après login")
        print(f"  Connecté en tant que : {user.get('name', '?')}")

        # --- 3. Téléchargement des PDF ---
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        ok_count = 0
        err_count = 0
        result_rows = []

        for bill in bills_with_owork:
            odoo_name = bill.get("name", "?")
            invoice_date = _sanitize(bill.get("invoice_date"))
            doc_id = int(bill["is_id_owork"])
            bill_id = str(bill["id"])
            t_dl = time.time()

            try:
                # Vérifier si un PDF avec ce préfixe existe déjà
                existing = list(OUTPUT_DIR.glob(f"{odoo_name}-*"))
                if existing:
                    elapsed = f"{time.time() - t_dl:.1f}s"
                    result_rows.append((bill_id, odoo_name, invoice_date, str(doc_id), elapsed, "✅", existing[0].name))
                    ok_count += 1
                    continue

                attach = owork_get_attachment(base_url, cookie_path, doc_id)
                if not attach:
                    elapsed = f"{time.time() - t_dl:.1f}s"
                    result_rows.append((bill_id, odoo_name, invoice_date, str(doc_id), elapsed, "⚠", "pas de pièce jointe"))
                    err_count += 1
                    continue

                attach_id = attach["id"]
                orig_filename = (
                    attach.get("datas_fname")
                    or attach.get("name")
                    or f"doc_{doc_id}.pdf"
                )
                # Préfixer avec le numéro de facture Odoo
                filename = f"{odoo_name}-{orig_filename}"
                output_path = OUTPUT_DIR / filename

                owork_download_pdf(base_url, cookie_path,
                                   attach_id, orig_filename, output_path)

                elapsed = f"{time.time() - t_dl:.1f}s"
                result_rows.append((bill_id, odoo_name, invoice_date, str(doc_id), elapsed, "📥", filename))
                ok_count += 1

            except Exception as exc:
                elapsed = f"{time.time() - t_dl:.1f}s"
                result_rows.append((bill_id, odoo_name, invoice_date, str(doc_id), elapsed, "❌", f"ERREUR: {exc}"))
                err_count += 1

        # Tableau récapitulatif
        print()
        print(_format_table(result_rows, ("ID", "Name", "Date", "is_id_owork", "Durée", " ", "PDF")))
        total = time.time() - t_start
        print(f"\nTerminé : {ok_count} téléchargé(s), {err_count} erreur(s) | Durée totale : {total:.1f}s")
        print(f"Dossier : {OUTPUT_DIR.resolve()}")

    finally:
        cookie_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)
