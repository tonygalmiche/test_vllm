#!/usr/bin/env python3
"""Login O'Work + récupération des infos d'un document + téléchargement du PDF."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from config import (
    OWORK_BASE_URL,
    OWORK_PASSWORD,
    OWORK_REALM,
    OWORK_USERNAME,
)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _ensure_curl_present() -> None:
    if shutil.which("curl") is None:
        raise SystemExit("curl est introuvable dans le PATH")


def _curl_json(cmd: list[str]) -> dict:
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


# ---------------------------------------------------------------------------
# Étape 1 – Login
# ---------------------------------------------------------------------------

def owork_login(base_url: str, username: str, password: str,
                realm: str, cookie_path: Path) -> dict:
    """POST /api/auth/user → renvoie le JSON utilisateur, enregistre le cookie jar."""
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


# ---------------------------------------------------------------------------
# Étape 2 – Récupération des infos du document (multiread)
# ---------------------------------------------------------------------------

def owork_get_doc_info(base_url: str, cookie_path: Path, doc_id: int) -> dict:
    """POST /api/db/multiread → renvoie les infos du document et de son attachment.

    Format attendu par O'Work (capturé via HAR) ::

        {"reads": {"<alias>": {"table": "<table>", "search": {...}, ...}}}

    On fait 2 lectures en un appel : le document et ses pièces jointes.
    """
    url = base_url.rstrip("/") + "/api/db/multiread"

    payload = json.dumps({
        "reads": {
            "doc": {
                "table": "iz_doc",
                "search": {"id": doc_id},
            },
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
    return _curl_json(cmd)


# ---------------------------------------------------------------------------
# Étape 3 – Téléchargement du PDF
# ---------------------------------------------------------------------------

def owork_download_pdf(base_url: str, cookie_path: Path,
                       attach_id: int, filename: str, output_path: Path) -> None:
    """GET /api/readBinary/download/<attach_id>/<filename> → enregistre le fichier."""
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
        raise RuntimeError(f"Téléchargement échoué (code {result.returncode})\n{result.stderr}")

    size = output_path.stat().st_size
    if size == 0:
        raise RuntimeError("Le fichier téléchargé est vide")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Se connecte à O'Work, récupère les infos d'un document "
            "et télécharge le PDF associé."
        )
    )
    parser.add_argument(
        "--base-url",
        default=OWORK_BASE_URL or "https://xxx-owork.fr",
        help="URL de la plateforme O'Work (défaut: %(default)s)",
    )

    if OWORK_USERNAME:
        parser.add_argument("--username", default=OWORK_USERNAME, help="Identifiant O'Work")
    else:
        parser.add_argument("--username", required=True, help="Identifiant O'Work")

    if OWORK_PASSWORD:
        parser.add_argument("--password", default=OWORK_PASSWORD, help="Mot de passe O'Work")
    else:
        parser.add_argument("--password", required=True, help="Mot de passe O'Work")

    parser.add_argument(
        "--realm", default=OWORK_REALM or "normal",
        help="Realm transmis à l'API (défaut: %(default)s)",
    )
    parser.add_argument(
        "--doc-id", type=int, default=2550,
        help="ID du document O'Work à télécharger (défaut: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Répertoire de destination du PDF (défaut: répertoire courant)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _ensure_curl_present()

    with tempfile.NamedTemporaryFile(prefix="owork_cookies_", delete=False) as tmp:
        cookie_path = Path(tmp.name)

    try:
        # --- 1. Login ---
        print(f"Connexion à {args.base_url} ...")
        user = owork_login(args.base_url, args.username, args.password,
                           args.realm, cookie_path)

        connect_sid = _extract_cookie(cookie_path, "connect.sid")
        if not connect_sid:
            raise RuntimeError("Cookie connect.sid introuvable après login")

        print(f"  Connecté en tant que : {user.get('name', user.get('login', '?'))}")
        print(f"  connect.sid = {connect_sid}")

        # --- 2. Récupération des infos du document ---
        print(f"\nRécupération du document #{args.doc_id} ...")
        data = owork_get_doc_info(args.base_url, cookie_path, args.doc_id)
        print(f"  Réponse multiread : {json.dumps(data, indent=2, ensure_ascii=False)[:2000]}")

        # Extraire le document
        doc_list = data.get("doc", [])
        if not doc_list:
            raise RuntimeError(f"Document #{args.doc_id} introuvable dans O'Work")
        doc = doc_list[0] if isinstance(doc_list, list) else doc_list

        # Extraire la pièce jointe
        attach_list = data.get("attach", [])
        if not attach_list:
            raise RuntimeError(
                f"Aucune pièce jointe trouvée pour le document #{args.doc_id}"
            )
        attach = attach_list[0] if isinstance(attach_list, list) else attach_list

        attach_id = attach["id"]
        filename = (
            attach.get("datas_fname")
            or attach.get("name")
            or f"doc_{args.doc_id}.pdf"
        )

        print(f"  Référence   : {doc.get('reference', doc.get('name', '?'))}")
        print(f"  Attachment  : id={attach_id}, fichier={filename}")

        # --- 3. Téléchargement du PDF ---
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        print(f"\nTéléchargement vers {output_path} ...")
        owork_download_pdf(args.base_url, cookie_path, attach_id, filename, output_path)

        size_kb = output_path.stat().st_size / 1024
        print(f"  Téléchargé avec succès ({size_kb:.1f} Ko)")

    finally:
        cookie_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)
