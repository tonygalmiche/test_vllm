#!/usr/bin/env python3
"""Analyse de factures *clients* (émises par InfoSaône) via vLLM."""

from __future__ import annotations

import json
from typing import Dict

from config import CLIENT_HINTS

from analyse_facture_common import (
    call_vllm_with_messages,
    is_emitter_name,
    normalize_identifier,
    run,
)


def build_prompt(pdf_text: str) -> str:
    """Construit l'invite envoyée au LLM pour une facture client."""

    client_hints_text = ", ".join(CLIENT_HINTS)
    instructions = (
        "Analyse l'extrait de facture ci-dessous et réponds uniquement par un JSON valide avec les clés\n"
        "numero, date, client, montant_ht, montant_tva, montant_ttc, identifiant_client.\n"
        "Repère et recopie exactement le nom du destinataire (client) tel qu'il apparaît sur la facture. Ignore les blocs correspondant à l'émetteur (logo, coordonnées de l'entreprise qui émet la facture). Inspecte le bloc d'adresse 'Facturé à', 'Client', ou la section contenant l'adresse du destinataire.\n"
        "Si aucune mention explicite du client n'est présente, laisse la valeur du champ client vide (\"\"). N'utilise jamais la chaîne 'Inconnu'.\n"
        "Ne renvoie jamais le nom de l'émetteur (ex: 'InfoSaône', 'Info Saone') ni celui de sa ville (ex: 'Pluvault') comme client.\n"
        "Pour identifiant_client, fournis soit le SIRET (14 chiffres) soit le SIREN (9 chiffres) du destinataire. Ne donne jamais les identifiants de l'émetteur ni plusieurs valeurs. Laisse vide si l'information est absente.\n"
        f"Clients habituels (liste indicative, à utiliser comme indice uniquement): {client_hints_text}.\n"
        "Respecte strictement la structure suivante: {\"numero\": \"...\", \"date\": \"...\", \"client\": \"...\", \"montant_ht\": \"...\", \"montant_tva\": \"...\", \"montant_ttc\": \"...\", \"identifiant_client\": \"...\"}.\n"
        "N'ajoute aucun texte avant ou après le JSON.\n"
        "Exemple attendu: {\"numero\": \"01181\", \"date\": \"01/02/2026\", \"client\": \"Plastigray\", \"montant_ht\": \"2375,00\", \"montant_tva\": \"475,00\", \"montant_ttc\": \"2850,00\", \"identifiant_client\": \"37784638100012\"}.\n"
    )

    return f"{instructions}\nFacture:\n'''\n{pdf_text}\n'''"


def ask_client_details(pdf_text: str) -> Dict[str, str]:
    """Effectue une requête ciblée pour identifier client et identifiant."""

    hints = ", ".join(CLIENT_HINTS)
    user_prompt = (
        "Analyse uniquement le destinataire (client) de la facture ci-dessous.\n"
        "L'émetteur est InfoSaône (ou variantes). Ne retourne ni InfoSaône ni sa ville.\n"
        "Donne un JSON avec les clés client et identifiant_client.\n"
        "identifiant_client doit être un SIRET (14 chiffres) ou un SIREN (9 chiffres). Laisse vide si absent.\n"
        f"Clients connus (indices, pas obligatoires): {hints}.\n"
        "Forme attendue: {\"client\": \"Nom\", \"identifiant_client\": \"123456789\"}.\n"
        "Facture:\n'''\n"
        f"{pdf_text}\n'''")

    messages = [
        {
            "role": "system",
            "content": "You extract the client/recipient name from invoices and answer only with JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]

    response = call_vllm_with_messages(messages, temperature=0.0)
    content = response["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}

    client_value = str(data.get("client", "")).strip()
    identifier_value = normalize_identifier(str(data.get("identifiant_client", "")))

    if client_value.lower() in {"inconnu", "unknown"}:
        client_value = ""
    if client_value and is_emitter_name(client_value):
        client_value = ""

    return {"client": client_value, "identifiant_client": identifier_value}


if __name__ == "__main__":  # pragma: no cover
    run(
        build_prompt_fn=build_prompt,
        ask_details_fn=ask_client_details,
        entity_hints=CLIENT_HINTS,
        entity_label="Client",
        entity_key="client",
        entity_id_key="identifiant_client",
    )


