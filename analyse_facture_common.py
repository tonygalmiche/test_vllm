"""Fonctions communes aux scripts d'analyse de factures via vLLM."""

from __future__ import annotations

import base64
import json
import ssl
import sys
import time
import unicodedata
import re
import xmlrpc.client
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from config import (
    API_KEY,
    BASE_URL,
    DEBUG,
    FACTURES_DIR,
    MIN_TEXT_LENGTH,
    MODEL_NAME,
    ODOO_DB,
    ODOO_PASSWORD,
    ODOO_URL,
    ODOO_USERNAME,
    ODOO_VERIFY_SSL,
    TEMPERATURE,
)

EMITTER_KEYWORDS = {
    "infosaône",
    "infosaone",
    "info saone",
    "info-saone",
}

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "La dépendance 'pypdf' est manquante. Installez-la avec 'pip install pypdf'."
    ) from exc

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Connexion Odoo XML-RPC
# ---------------------------------------------------------------------------

def _build_odoo_transport() -> Optional[xmlrpc.client.Transport]:
    """Retourne un transport XML-RPC qui ignore SSL si besoin."""
    if ODOO_URL.lower().startswith("https://") and not ODOO_VERIFY_SSL:
        class _UnsafeTransport(xmlrpc.client.SafeTransport):
            def __init__(self) -> None:
                super().__init__()
                self.context = ssl._create_unverified_context()
        return _UnsafeTransport()
    return None


_ODOO_TRANSPORT = _build_odoo_transport()


def odoo_connect() -> tuple[int, xmlrpc.client.ServerProxy]:
    """Se connecte à Odoo via XML-RPC et retourne (uid, models_proxy)."""
    common = xmlrpc.client.ServerProxy(
        f"{ODOO_URL}/xmlrpc/2/common",
        allow_none=True,
        transport=_ODOO_TRANSPORT,
    )
    uid = common.authenticate(ODOO_DB, ODOO_USERNAME, ODOO_PASSWORD, {})
    if not uid:
        raise RuntimeError("Authentification Odoo échouée: vérifiez config.py")

    models = xmlrpc.client.ServerProxy(
        f"{ODOO_URL}/xmlrpc/2/object",
        allow_none=True,
        transport=_ODOO_TRANSPORT,
    )
    return uid, models


def odoo_fetch_vendor_bills_by_names(
    names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Récupère les factures fournisseur Odoo dont le ``name`` est dans *names*.

    Retourne un dict indexé par ``name`` avec les champs utiles à la comparaison.
    """
    if not names:
        return {}

    uid, models = odoo_connect()

    domain = [
        ("move_type", "=", "in_invoice"),
        ("name", "in", names),
    ]

    bills = models.execute_kw(
        ODOO_DB, uid, ODOO_PASSWORD,
        "account.move", "search_read",
        [domain],
        {
            "fields": [
                "name", "ref", "invoice_date",
                "partner_id",
                "amount_untaxed", "amount_tax", "amount_total",
            ],
        },
    )

    result: Dict[str, Dict[str, Any]] = {}
    for bill in bills:
        odoo_name = bill.get("name", "")
        partner = bill.get("partner_id")
        partner_name = partner[1] if isinstance(partner, (list, tuple)) and len(partner) > 1 else ""
        result[odoo_name] = {
            "name": odoo_name,
            "ref": bill.get("ref") or "",
            "invoice_date": bill.get("invoice_date") or "",
            "partner_name": partner_name,
            "amount_untaxed": bill.get("amount_untaxed", 0.0),
            "amount_tax": bill.get("amount_tax", 0.0),
            "amount_total": bill.get("amount_total", 0.0),
        }
    return result


def _extract_odoo_name_from_filename(filename: str) -> str:
    """Extrait le name Odoo depuis un nom de fichier ``{odoo_name}-{reste}.PDF``."""
    # Le name Odoo est la partie avant le premier tiret
    # Ex: "127987-3260262257.PDF" → "127987"
    parts = filename.split("-", 1)
    return parts[0] if parts else ""


def compare_with_odoo(
    extraction: Dict[str, Any],
    odoo_bill: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare les 3 montants (HT, TVA, TTC) entre extraction PDF et Odoo.

    Retourne un dict avec les résultats de comparaison par champ
    et les valeurs détaillées pour le debug.
    """
    checks: Dict[str, Any] = {}

    # --- Montant HT ---
    pdf_ht = parse_decimal(extraction.get("montant_ht"))
    odoo_ht = Decimal(str(odoo_bill.get("amount_untaxed", 0)))
    if pdf_ht is not None:
        checks["ht"] = "✅" if abs(pdf_ht - odoo_ht) < Decimal("0.05") else "❌"
    else:
        checks["ht"] = "❓"
    checks["ht_detail"] = {"pdf": pdf_ht, "odoo": odoo_ht}

    # --- Montant TVA ---
    pdf_tva = parse_decimal(extraction.get("montant_tva"))
    odoo_tva = Decimal(str(odoo_bill.get("amount_tax", 0)))
    if pdf_tva is not None:
        checks["tva"] = "✅" if abs(pdf_tva - odoo_tva) < Decimal("0.05") else "❌"
    else:
        checks["tva"] = "❓"
    checks["tva_detail"] = {"pdf": pdf_tva, "odoo": odoo_tva}

    # --- Montant TTC ---
    pdf_ttc = parse_decimal(extraction.get("montant_ttc"))
    odoo_ttc = Decimal(str(odoo_bill.get("amount_total", 0)))
    if pdf_ttc is not None:
        checks["ttc"] = "✅" if abs(pdf_ttc - odoo_ttc) < Decimal("0.05") else "❌"
    else:
        checks["ttc"] = "❓"
    checks["ttc_detail"] = {"pdf": pdf_ttc, "odoo": odoo_ttc}

    # --- Synthèse globale ---
    status_keys = ["ht", "tva", "ttc"]
    if all(checks[k] == "✅" for k in status_keys):
        checks["global"] = "✅"
    elif any(checks[k] == "❌" for k in status_keys):
        checks["global"] = "❌"
    else:
        checks["global"] = "❓"

    return checks


# ---------------------------------------------------------------------------
# Utilitaires PDF
# ---------------------------------------------------------------------------

def list_pdf_files(root: Path) -> List[Path]:
    """Retourne l'ensemble des PDF présents dans *root* (sans récursivité)."""

    if not root.exists():
        raise FileNotFoundError(f"Missing invoices folder: {root}")

    pdfs = sorted(p for p in root.iterdir() if p.suffix.lower() == ".pdf")

    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {root}")

    return pdfs


def extract_text(pdf_path: Path) -> str:
    """Extrait le texte brut d'un PDF via pypdf."""

    reader = PdfReader(str(pdf_path))
    chunks: List[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        content = page.extract_text() or ""
        chunks.append(f"--- Page {page_number} ---\n{content.strip()}\n")

    return "\n".join(chunks).strip()


def _text_is_usable(pdf_text: str) -> bool:
    """Vérifie si le texte extrait contient assez de contenu exploitable."""
    # Retirer les en-têtes "--- Page N ---" pour compter le vrai contenu
    clean = re.sub(r"---\s*Page\s*\d+\s*---", "", pdf_text).strip()
    return len(clean) >= MIN_TEXT_LENGTH


def pdf_to_base64_images(pdf_path: Path, dpi: int = 200, max_pages: int = 0) -> List[str]:
    """Convertit chaque page d'un PDF en image PNG encodée en base64.

    Parameters
    ----------
    max_pages : int
        Nombre maximal de pages à convertir (0 = toutes).
    """

    if fitz is None:
        raise SystemExit(
            "La dépendance 'pymupdf' est manquante. Installez-la avec 'pip install pymupdf'."
        )

    doc = fitz.open(str(pdf_path))
    images: List[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        images.append(b64)

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Communication avec le serveur vLLM
# ---------------------------------------------------------------------------

def call_vllm_with_messages(messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
    """Envoie des messages arbitraires à l'API HTTP vLLM."""

    endpoint = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def call_vllm(prompt: str, temperature: float = TEMPERATURE) -> Dict[str, Any]:
    """Envoie l'invite standard à vLLM et renvoie la réponse complète."""

    messages = [
        {
            "role": "system",
            "content": "You extract structured invoice data and answer strictly in JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    return call_vllm_with_messages(messages, temperature)


def call_vllm_vision(
    prompt: str, images_b64: List[str], temperature: float = TEMPERATURE
) -> Dict[str, Any]:
    """Envoie des images (base64) + un prompt texte au LLM via l'API vision."""

    # Construire le contenu multimodal : images puis texte
    content: List[Dict[str, Any]] = []
    for img_b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "system",
            "content": "You extract structured invoice data from images and answer strictly in JSON.",
        },
        {"role": "user", "content": content},
    ]
    return call_vllm_with_messages(messages, temperature)


def _ask_entity_vision(
    pdf_path: Path,
    entity_label: str,
    entity_key: str,
    entity_id_key: str,
    exclude_names: List[str] | None = None,
) -> Dict[str, str]:
    """Requête vision ciblée pour trouver uniquement le nom de l'entité.

    Utile quand le nom du fournisseur est dans un logo/image et que le texte
    extrait ne le contient pas.
    """

    exclude_text = ""
    if exclude_names:
        exclude_text = (
            f"\nATTENTION : les noms suivants sont ceux du DESTINATAIRE, "
            f"pas du {entity_label.lower()}. Ne les retourne JAMAIS comme {entity_label.lower()} : "
            + ", ".join(exclude_names)
        )

    prompt = (
        f"Regarde cette facture. Identifie le nom du {entity_label.lower()} "
        f"(celui qui ÉMET la facture, dont le nom/logo apparaît en haut).\n"
        f"Identifie aussi son SIRET (14 chiffres) ou SIREN (9 chiffres) de l'ÉMETTEUR. "
        f"Cherche en priorité une mention 'SIREN=' ou 'SIRET=' sur la facture. "
        f"Ne retourne PAS le numéro de TVA intracommunautaire (commençant par FR). "
        f"ATTENTION : l'identifiant doit être celui de l'ÉMETTEUR, "
        f"pas celui du destinataire.\n"
        f"{exclude_text}\n\n"
        f"Réponds UNIQUEMENT en JSON : {{\"{entity_key}\": \"...\", \"{entity_id_key}\": \"...\"}}\n"
        f"Si tu ne trouves pas le SIRET ou SIREN, laisse une chaîne vide."
    )

    images_b64 = pdf_to_base64_images(pdf_path, max_pages=1)
    api_response = call_vllm_vision(prompt, images_b64, temperature=TEMPERATURE)
    content = api_response["choices"][0]["message"]["content"].strip()

    if DEBUG:
        print(f"  🖼 Vision entity response: {content[:300]}", file=sys.stderr)

    parsed = parse_llm_json(content)
    result: Dict[str, str] = {}

    entity_value = str(parsed.get(entity_key, "")).strip()
    if entity_value and entity_value.lower() not in {"inconnu", "unknown"}:
        # Vérifier exclusions
        excluded = False
        if is_emitter_name(entity_value):
            excluded = True
        if not excluded and exclude_names:
            norm = normalize_for_matching(entity_value)
            for excl in exclude_names:
                if normalize_for_matching(excl) in norm or norm in normalize_for_matching(excl):
                    excluded = True
                    break
        if not excluded:
            result[entity_key] = entity_value

    id_value = str(parsed.get(entity_id_key, "")).strip()
    if id_value and id_value.lower() not in {"inconnu", "unknown"}:
        result[entity_id_key] = normalize_identifier(id_value)

    return result


# ---------------------------------------------------------------------------
# Parsing et nettoyage
# ---------------------------------------------------------------------------

def parse_llm_json(raw_content: str) -> Dict[str, Any]:
    """Tente de parser la réponse du LLM en JSON, sinon retourne le contenu brut."""

    text = raw_content.strip()

    # Gérer les réponses enveloppées dans un bloc markdown ```json ... ```
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": raw_content}


def normalize_for_matching(text: str) -> str:
    """Normalise une chaîne pour faciliter les comparaisons insensibles à la casse."""

    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.lower()


def is_emitter_name(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return any(keyword in normalized for keyword in EMITTER_KEYWORDS)


def normalize_identifier(value: str) -> str:
    """Normalise un identifiant entreprise (SIRET 14, SIREN 9, TVA intra 11→SIREN)."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 14:
        return digits
    if len(digits) == 9:
        return digits
    # TVA intracommunautaire FR : 2 chiffres clé + SIREN 9 chiffres = 11
    if len(digits) == 11:
        return digits[2:]  # extraire le SIREN (9 derniers chiffres)
    return ""


def sanitize_extraction(
    data: Dict[str, Any],
    entity_key: str = "client",
    entity_id_key: str = "identifiant_client",
    exclude_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Nettoie les champs extraits (espaces, valeurs 'Inconnu', etc.)."""

    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = value.strip()
        else:
            sanitized[key] = value

    entity_value = str(sanitized.get(entity_key, "")).strip()
    entity_excluded = False
    if entity_value.lower() in {"inconnu", "unknown"}:
        entity_value = ""
    if entity_value and is_emitter_name(entity_value):
        entity_value = ""
        entity_excluded = True
    if entity_value and exclude_names:
        norm = normalize_for_matching(entity_value)
        for excl in exclude_names:
            excl_norm = normalize_for_matching(excl)
            if excl_norm in norm or norm in excl_norm:
                entity_value = ""
                entity_excluded = True
                break
    sanitized[entity_key] = entity_value

    identifier_value = str(sanitized.get(entity_id_key, "")).strip()
    # Si l'entité a été exclue, l'identifiant lui appartient : on le vide aussi
    if entity_excluded:
        sanitized[entity_id_key] = ""
    else:
        sanitized[entity_id_key] = normalize_identifier(identifier_value)

    return sanitized


def guess_entity_from_text(pdf_text: str, hints: List[str]) -> Optional[str]:
    """Tente de retrouver une entité connue en inspectant le texte extrait du PDF."""

    normalized_text = normalize_for_matching(pdf_text)
    for hint in hints:
        normalized_hint = normalize_for_matching(hint)
        if normalized_hint and normalized_hint in normalized_text:
            if not is_emitter_name(hint):
                return hint
    return None


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def display_field(value: Any) -> str:
    """Retourne la valeur prête pour affichage (croix rouge si vide)."""

    text = str(value or "").strip()
    return text if text else "❌"


def visual_width(text: str) -> int:
    """Calcule la largeur affichée d'une chaîne en tenant compte des caractères larges."""

    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        kind = unicodedata.east_asian_width(char)
        width += 2 if kind in {"W", "F"} else 1
    return width


def pad_cell(value: str, target_width: int) -> str:
    """Complète une cellule avec des espaces pour atteindre la largeur souhaitée."""

    padding = max(target_width - visual_width(value), 0)
    return value + (" " * padding)


def parse_decimal(value: Any) -> Decimal | None:
    """Convertit une valeur texte ou numérique en Decimal."""

    if value is None:
        return None

    if isinstance(value, Decimal):
        return value

    text = str(value).strip()
    if not text:
        return None

    normalized = (
        text.replace("€", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace(",", ".")
    )

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _is_missing(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    normalized = text.lower()
    return normalized in {"inconnu", "unknown"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(data: Dict[str, Any], entity_key: str = "client", entity_id_key: str = "identifiant_client") -> tuple[bool, str]:
    """Contrôle la complétude des champs et la cohérence des montants."""

    required_keys = [
        "numero",
        "date",
        entity_key,
        entity_id_key,
        "montant_ht",
        "montant_tva",
        "montant_ttc",
    ]
    missing = [key for key in required_keys if _is_missing(data.get(key))]
    if missing:
        return False, f"Champs manquants ou 'Inconnu': {', '.join(missing)}"

    amounts = {key: parse_decimal(data.get(key)) for key in ("montant_ht", "montant_tva", "montant_ttc")}
    invalid_amounts = [key for key, value in amounts.items() if value is None]
    if invalid_amounts:
        return False, f"Montants invalides: {', '.join(invalid_amounts)}"

    ht = amounts["montant_ht"]
    tva = amounts["montant_tva"]
    ttc = amounts["montant_ttc"]
    if abs((ht + tva) - ttc) > Decimal("0.05"):
        return False, "Incohérence HT + TVA != TTC"

    return True, ""


# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------

def render_summary_table(
    results: List[Dict[str, Any]],
    total_seconds: float,
    entity_label: str = "Client",
    entity_key: str = "client",
    entity_id_key: str = "identifiant_client",
    odoo_bills: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Affiche un tableau formaté avec les champs principaux et les durées."""

    if not results:
        return

    has_odoo = odoo_bills is not None and len(odoo_bills) > 0

    headers = [
        "Fichier",
        "Numéro",
        "Date",
        entity_label,
        "Identifiant",
        "HT",
        "TVA",
        "TTC",
        "Durée (s)",
        "Validation",
    ]
    if has_odoo:
        headers.append("Odoo")

    anomaly_keys = [
        "numero",
        "date",
        entity_key,
        entity_id_key,
        "montant_ht",
        "montant_tva",
        "montant_ttc",
    ]

    rows: List[List[str]] = []
    total_anomalies = 0
    invoices_with_anomalies = 0
    for entry in results:
        extraction = entry.get("extraction", {}) or {}
        invoice_anomalies = sum(1 for key in anomaly_keys if _is_missing(extraction.get(key)))
        total_anomalies += invoice_anomalies
        if invoice_anomalies:
            invoices_with_anomalies += 1

        rows.append(
            [
                entry.get("file", ""),
                display_field(extraction.get("numero")),
                display_field(extraction.get("date")),
                display_field(extraction.get(entity_key)),
                display_field(extraction.get(entity_id_key)),
                display_field(extraction.get("montant_ht")),
                display_field(extraction.get("montant_tva")),
                display_field(extraction.get("montant_ttc")),
                f"{entry.get('duration_s', 0.0):.2f}",
                "✅" if entry.get("status_ok") else "❌",
            ]
        )
        if has_odoo:
            odoo_check = entry.get("odoo_check", {})
            odoo_global = odoo_check.get("global", "—")
            ht_check = odoo_check.get("ht", "")
            tva_check = odoo_check.get("tva", "")
            ttc_check = odoo_check.get("ttc", "")
            detail = f"{odoo_global} HT:{ht_check} TVA:{tva_check} TTC:{ttc_check}"
            rows[-1].append(detail)

    widths = [visual_width(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], visual_width(value))

    def format_row(values: List[str]) -> str:
        padded_values = [pad_cell(value, widths[idx]) for idx, value in enumerate(values)]
        return " | ".join(padded_values)

    separator = "-+-".join("-" * width for width in widths)

    print("\nTableau de synthèse:", file=sys.stderr)
    print(format_row(headers), file=sys.stderr)
    print(separator, file=sys.stderr)
    for row in rows:
        print(format_row(row), file=sys.stderr)

    count = len(results)
    avg = total_seconds / count if count else 0.0
    print(
        f"\nDurée totale: {total_seconds:.2f} s | Factures: {count} | Moyenne: {avg:.2f} s",
        file=sys.stderr,
    )

    if count:
        percent_with_anomaly = (invoices_with_anomalies / count) * 100
    else:
        percent_with_anomaly = 0.0

    print(
        f"Total d'anomalies (❌): {total_anomalies} | Factures avec anomalie: {invoices_with_anomalies} | Pourcentage: {percent_with_anomaly:.2f}%",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Boucle principale paramétrable
# ---------------------------------------------------------------------------

def run(
    build_prompt_fn: Callable[[str], str],
    ask_details_fn: Callable[[str], Dict[str, str]],
    factures_dir: Path | None = None,
    entity_hints: List[str] | None = None,
    entity_label: str = "Client",
    entity_key: str = "client",
    entity_id_key: str = "identifiant_client",
    exclude_names: List[str] | None = None,
    check_odoo: bool = False,
) -> None:
    """Boucle principale d'analyse de factures.

    Paramètres
    ----------
    build_prompt_fn : callable
        Construit le prompt principal à partir du texte PDF.
    ask_details_fn : callable
        Effectue une requête ciblée pour identifier l'entité et son identifiant.
    factures_dir : Path, optional
        Dossier contenant les PDF. Par défaut ``FACTURES_DIR``.
    entity_hints : list[str], optional
        Noms connus pour deviner l'entité dans le texte.
    entity_label : str
        Libellé affiché dans le tableau (ex: "Client", "Fournisseur").
    entity_key : str
        Clé JSON de l'entité (ex: "client", "fournisseur").
    entity_id_key : str
        Clé JSON de l'identifiant (ex: "identifiant_client", "identifiant_fournisseur").
    exclude_names : list[str], optional
        Noms à exclure du champ entité (ex: noms des destinataires).
    check_odoo : bool
        Si True, vérifie la concordance avec les données Odoo.
    """

    invoices_dir = Path(factures_dir or FACTURES_DIR).expanduser().resolve()

    try:
        pdf_files = list_pdf_files(invoices_dir)
    except FileNotFoundError as err:
        raise SystemExit(str(err)) from err

    # --- Récupération des factures Odoo pour comparaison ---
    odoo_bills: Dict[str, Dict[str, Any]] = {}
    if check_odoo:
        odoo_names = [_extract_odoo_name_from_filename(p.name) for p in pdf_files]
        odoo_names = [n for n in odoo_names if n]
        if odoo_names:
            print(f"Récupération des factures Odoo ({len(odoo_names)} noms)…", file=sys.stderr)
            try:
                odoo_bills = odoo_fetch_vendor_bills_by_names(odoo_names)
                print(f"  {len(odoo_bills)} facture(s) trouvée(s) dans Odoo", file=sys.stderr)
            except Exception as exc:
                print(f"  ⚠ Erreur Odoo: {exc}", file=sys.stderr)

    results = []
    total_start = time.perf_counter()

    for pdf_path in pdf_files:
        file_start = time.perf_counter()
        print(f"Processing {pdf_path.name}…", file=sys.stderr)
        pdf_text = extract_text(pdf_path)

        if DEBUG:
            print(f"  Texte extrait: {len(pdf_text)} caractères", file=sys.stderr)
            if len(pdf_text) < 50:
                print(f"  ⚠ Texte quasi vide: {pdf_text!r}", file=sys.stderr)
            else:
                print(f"  Aperçu: {pdf_text[:200]!r}…", file=sys.stderr)

        # Déterminer le mode : texte ou vision
        use_vision = not _text_is_usable(pdf_text)

        if use_vision:
            print(f"  🖼 Mode vision activé (texte insuffisant)", file=sys.stderr)
            images_b64 = pdf_to_base64_images(pdf_path)
            prompt = build_prompt_fn("")  # prompt sans texte PDF
            api_response = call_vllm_vision(prompt, images_b64, temperature=TEMPERATURE)
        else:
            prompt = build_prompt_fn(pdf_text)
            api_response = call_vllm(prompt, temperature=TEMPERATURE)

        content = api_response["choices"][0]["message"]["content"].strip()

        if DEBUG:
            print(f"  Réponse LLM: {content[:300]}", file=sys.stderr)

        parsed_json = sanitize_extraction(parse_llm_json(content), entity_key, entity_id_key, exclude_names=exclude_names)

        needs_entity = not parsed_json.get(entity_key)
        needs_identifier = not parsed_json.get(entity_id_key)

        if needs_entity or needs_identifier:
            focused_details = ask_details_fn(pdf_text)
            focused_entity = focused_details.get(entity_key)
            focused_identifier = focused_details.get(entity_id_key)

            if needs_entity and focused_entity:
                parsed_json[entity_key] = focused_entity
                print(
                    f"Info: {entity_label.lower()} trouvé via requête dédiée ({focused_entity}) pour {pdf_path.name}.",
                    file=sys.stderr,
                )

            if needs_identifier and focused_identifier:
                # N'accepter l'identifiant QUE si on a aussi trouvé l'entité
                # (sinon l'identifiant appartient probablement à l'entité exclue)
                if parsed_json.get(entity_key):
                    parsed_json[entity_id_key] = focused_identifier
                    print(
                        f"Info: identifiant {entity_label.lower()} trouvé via requête dédiée ({focused_identifier}) pour {pdf_path.name}.",
                        file=sys.stderr,
                    )
                else:
                    if DEBUG:
                        print(
                            f"  ⚠ Identifiant ignoré ({focused_identifier}) : entité encore inconnue",
                            file=sys.stderr,
                        )

        if not parsed_json.get(entity_key) and entity_hints:
            guessed = guess_entity_from_text(pdf_text, entity_hints)
            if guessed:
                parsed_json[entity_key] = guessed
                print(
                    f"Info: {entity_label.lower()} déduit automatiquement ({guessed}) pour {pdf_path.name}.",
                    file=sys.stderr,
                )

        # -- Fallback vision : déclenché quand l'entité est manquante OU
        #    quand elle semble suspecte (absente de l'en-tête du texte,
        #    ce qui signifie qu'elle a été prise dans le corps / lignes d'articles).
        entity_found = parsed_json.get(entity_key, "")
        entity_suspect = False
        if entity_found and not use_vision:
            # Vérifier si le fournisseur trouvé apparaît dans les ~300 premiers
            # caractères du texte (zone en-tête). Sinon il vient probablement
            # du corps de la facture (nom d'article, désignation, etc.).
            header_zone = normalize_for_matching(pdf_text[:300])
            entity_norm = normalize_for_matching(entity_found)
            if entity_norm not in header_zone:
                entity_suspect = True
                if DEBUG:
                    print(
                        f"  ⚠ {entity_label} '{entity_found}' absent de l'en-tête du texte → suspect",
                        file=sys.stderr,
                    )

        if (not entity_found or entity_suspect) and not use_vision:
            print(
                f"  🖼 Tentative vision ciblée pour {entity_label.lower()} de {pdf_path.name}…",
                file=sys.stderr,
            )
            vision_details = _ask_entity_vision(
                pdf_path, entity_label, entity_key, entity_id_key,
                exclude_names=exclude_names,
            )
            vision_entity = vision_details.get(entity_key)
            if vision_entity:
                parsed_json[entity_key] = vision_entity
                print(
                    f"  ✅ {entity_label} trouvé via vision : {vision_entity}",
                    file=sys.stderr,
                )
                # L'identifiant précédent appartenait à l'entité exclue :
                # on le remplace par celui de la vision (même vide)
                vision_id = vision_details.get(entity_id_key, "")
                parsed_json[entity_id_key] = vision_id
                if vision_id:
                    print(
                        f"  ✅ Identifiant {entity_label.lower()} trouvé via vision : {vision_id}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  ⚠ Identifiant {entity_label.lower()} non trouvé via vision",
                        file=sys.stderr,
                    )

        status_ok, status_msg = validate_extraction(parsed_json, entity_key, entity_id_key)
        duration = time.perf_counter() - file_start

        if not status_ok:
            print(
                f"Avertissement: extraction incomplète pour {pdf_path.name} ({status_msg}).",
                file=sys.stderr,
            )

        results.append(
            {
                "file": pdf_path.name,
                "extraction": parsed_json,
                "raw_response": api_response,
                "duration_s": duration,
                "status_ok": status_ok,
                "status_message": status_msg,
            }
        )

        # --- Comparaison avec Odoo ---
        if check_odoo and odoo_bills:
            odoo_name = _extract_odoo_name_from_filename(pdf_path.name)
            odoo_bill = odoo_bills.get(odoo_name)
            if odoo_bill:
                odoo_check = compare_with_odoo(parsed_json, odoo_bill)
                results[-1]["odoo_check"] = odoo_check
                if DEBUG:
                    ht_d = odoo_check.get("ht_detail", {})
                    tva_d = odoo_check.get("tva_detail", {})
                    ttc_d = odoo_check.get("ttc_detail", {})
                    print(
                        f"  Odoo: HT: PDF={ht_d.get('pdf')} vs Odoo={ht_d.get('odoo')} {odoo_check.get('ht','')}"
                        f" | TVA: PDF={tva_d.get('pdf')} vs Odoo={tva_d.get('odoo')} {odoo_check.get('tva','')}"
                        f" | TTC: PDF={ttc_d.get('pdf')} vs Odoo={ttc_d.get('odoo')} {odoo_check.get('ttc','')}"
                        f" → {odoo_check.get('global', '?')}",
                        file=sys.stderr,
                    )
            else:
                results[-1]["odoo_check"] = {"global": "—"}
                if DEBUG:
                    print(f"  ⚠ Facture Odoo '{odoo_name}' non trouvée", file=sys.stderr)

    total_duration = time.perf_counter() - total_start
    render_summary_table(
        results, total_duration, entity_label, entity_key, entity_id_key,
        odoo_bills=odoo_bills if check_odoo else None,
    )
