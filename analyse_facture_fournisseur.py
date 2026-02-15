#!/usr/bin/env python3
"""Analyse de factures *fournisseurs* (reçues par InfoSaône) via vLLM."""

from __future__ import annotations

import json
from typing import Dict

from config import FOURNISSEUR_HINTS, FOURNISSEUR_EXCLUDE, FACTURES_FOURNISSEUR_DIR

from analyse_facture_common import (
    call_vllm_with_messages,
    is_emitter_name,
    normalize_for_matching,
    normalize_identifier,
    run,
)


def _is_excluded_name(name: str) -> bool:
    """Vérifie si le nom correspond à un destinataire à exclure."""
    normalized = normalize_for_matching(name)
    return any(
        normalize_for_matching(excl) in normalized or normalized in normalize_for_matching(excl)
        for excl in FOURNISSEUR_EXCLUDE
    ) or is_emitter_name(name)


def build_prompt(pdf_text: str) -> str:
    """Construit l'invite envoyée au LLM pour une facture fournisseur."""

    fournisseur_hints_text = ", ".join(FOURNISSEUR_HINTS)
    exclude_text = ", ".join(f"'{n}'" for n in FOURNISSEUR_EXCLUDE)
    instructions = (
        "Analyse l'extrait de facture fournisseur ci-dessous et réponds uniquement par un JSON valide avec les clés\n"
        "numero, date, fournisseur, montant_ht, montant_tva, montant_ttc, identifiant_fournisseur.\n\n"
        "RÈGLES POUR IDENTIFIER LE FOURNISSEUR :\n"
        "- Le fournisseur est l'entreprise qui ÉMET la facture.\n"
        "- Son nom figure UNIQUEMENT dans l'en-tête ou le logo en haut de la facture, "
        "accompagné de son adresse, SIRET/SIREN ou numéro de TVA.\n"
        "- Le fournisseur ne se trouve JAMAIS dans le détail des lignes de la facture "
        "(codes articles, désignations de produits, références, quantités, prix unitaires).\n"
        "- Ne confonds JAMAIS le fournisseur avec un nom d'article, de produit, de matière "
        "ou de désignation figurant dans les lignes de détail.\n"
        "- Si le nom du fournisseur n'est pas clairement visible dans l'en-tête du texte, "
        "laisse le champ fournisseur vide (\"\").\n\n"
        f"Le destinataire (client / facturé à) est InfoSaône, Plastigray ou une de leurs filiales : ignore-le.\n"
        f"Ne renvoie JAMAIS un de ces noms comme fournisseur (ce sont les destinataires) : {exclude_text}.\n"
        "Pour identifiant_fournisseur, fournis soit le SIRET (14 chiffres) soit le SIREN (9 chiffres) de l'émetteur. "
        "Ne donne jamais les identifiants du destinataire ni plusieurs valeurs. Laisse vide si l'information est absente.\n"
        f"Fournisseurs habituels (liste indicative, à utiliser comme indice uniquement): {fournisseur_hints_text}.\n"
        "Respecte strictement la structure suivante: {{\"numero\": \"...\", \"date\": \"...\", \"fournisseur\": \"...\", \"montant_ht\": \"...\", \"montant_tva\": \"...\", \"montant_ttc\": \"...\", \"identifiant_fournisseur\": \"...\"}}.\n"
        "N'ajoute aucun texte avant ou après le JSON.\n"
    )

    return f"{instructions}\nFacture:\n'''\n{pdf_text}\n'''"


def ask_fournisseur_details(pdf_text: str) -> Dict[str, str]:
    """Effectue une requête ciblée pour identifier le fournisseur et son identifiant."""

    hints = ", ".join(FOURNISSEUR_HINTS)
    exclude_text = ", ".join(f"'{n}'" for n in FOURNISSEUR_EXCLUDE)
    user_prompt = (
        "Analyse uniquement l'émetteur (fournisseur) de la facture ci-dessous.\n"
        f"Le destinataire est un de ces noms (ne les retourne JAMAIS comme fournisseur) : {exclude_text}.\n"
        "Donne un JSON avec les clés fournisseur et identifiant_fournisseur.\n"
        "identifiant_fournisseur doit être un SIRET (14 chiffres) ou un SIREN (9 chiffres). Laisse vide si absent.\n"
        f"Fournisseurs connus (indices, pas obligatoires): {hints}.\n"
        "Forme attendue: {{\"fournisseur\": \"Nom\", \"identifiant_fournisseur\": \"123456789\"}}.\n"
        "Facture:\n'''\n"
        f"{pdf_text}\n'''")

    messages = [
        {
            "role": "system",
            "content": "You extract the supplier/vendor name from invoices and answer only with JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]

    response = call_vllm_with_messages(messages, temperature=0.0)
    content = response["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}

    fournisseur_value = str(data.get("fournisseur", "")).strip()
    identifier_value = normalize_identifier(str(data.get("identifiant_fournisseur", "")))

    if fournisseur_value.lower() in {"inconnu", "unknown"}:
        fournisseur_value = ""
    if fournisseur_value and _is_excluded_name(fournisseur_value):
        fournisseur_value = ""

    return {"fournisseur": fournisseur_value, "identifiant_fournisseur": identifier_value}


if __name__ == "__main__":  # pragma: no cover
    run(
        build_prompt_fn=build_prompt,
        ask_details_fn=ask_fournisseur_details,
        factures_dir=FACTURES_FOURNISSEUR_DIR,
        entity_hints=FOURNISSEUR_HINTS,
        entity_label="Fournisseur",
        entity_key="fournisseur",
        entity_id_key="identifiant_fournisseur",
        exclude_names=FOURNISSEUR_EXCLUDE,
        check_odoo=True,
    )
