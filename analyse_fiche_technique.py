#!/usr/bin/env python3
"""Analyse de fiches techniques via vLLM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from config import (
    FICHES_TECHNIQUES_DIR,
    FICHES_TECHNIQUES_PARAMS,
    FOURNISSEUR_EXCLUDE,
    LLM_PROVIDER,
    GEMINI_MODEL,
    DEEPINFRA_MODEL,
    MODEL_NAME,
)

from analyse_facture_common import (
    call_vllm_with_messages,
    call_vllm,
    call_vllm_vision,
    extract_text,
    is_emitter_name,
    list_pdf_files,
    normalize_for_matching,
    parse_llm_json,
    pdf_to_base64_images,
)


def _is_excluded_name(name: str) -> bool:
    """Vérifie si le nom correspond à un destinataire à exclure."""
    normalized = normalize_for_matching(name)
    return any(
        normalize_for_matching(excl) in normalized or normalized in normalize_for_matching(excl)
        for excl in FOURNISSEUR_EXCLUDE
    ) or is_emitter_name(name)


def build_prompt(pdf_text: str) -> str:
    """Construit l'invite envoyée au LLM pour une fiche technique."""

    exclude_text = ", ".join(f"'{n}'" for n in FOURNISSEUR_EXCLUDE)
    param_keys = ", ".join(FICHES_TECHNIQUES_PARAMS.keys())
    
    # Construction des règles pour chaque paramètre
    rules_text = ""
    for param_name, param_prompt in FICHES_TECHNIQUES_PARAMS.items():
        rules_text += f"\nRÈGLES POUR IDENTIFIER {param_name.upper().replace('_', ' ')} :\n"
        rules_text += f"- {param_prompt}\n"
    
    # Construction de la structure JSON attendue
    json_structure = "{" + ", ".join(f'"{k}": "..."' for k in FICHES_TECHNIQUES_PARAMS.keys()) + "}"
    
    instructions = (
        f"Analyse l'extrait de fiche technique ci-dessous et réponds uniquement par un JSON valide avec les clés\n"
        f"{param_keys}.\n"
        f"{rules_text}\n"
        f"Ne renvoie JAMAIS un de ces noms comme fournisseur (ce sont nos entreprises) : {exclude_text}.\n"
        f"Respecte strictement la structure suivante: {json_structure}.\n"
        "N'ajoute aucun texte avant ou après le JSON.\n"
    )

    return f"{instructions}\nFiche technique:\n'''\n{pdf_text}\n'''"


def ask_fiche_technique_details(pdf_text: str) -> Dict[str, str]:
    """Effectue une requête ciblée pour identifier tous les paramètres de la fiche technique."""

    exclude_text = ", ".join(f"'{n}'" for n in FOURNISSEUR_EXCLUDE)
    param_keys = ", ".join(FICHES_TECHNIQUES_PARAMS.keys())
    
    # Construction des descriptions pour chaque paramètre
    param_descriptions = ""
    for param_name, param_prompt in FICHES_TECHNIQUES_PARAMS.items():
        param_descriptions += f"{param_name}: {param_prompt}\n"
    
    # Construction de la structure JSON attendue
    json_structure = "{" + ", ".join(f'"{k}": "..."' for k in FICHES_TECHNIQUES_PARAMS.keys()) + "}"
    
    user_prompt = (
        f"Analyse la fiche technique ci-dessous et extrait les informations suivantes : {param_keys}.\n"
        f"Ne retourne JAMAIS un de ces noms comme fournisseur : {exclude_text}.\n"
        f"Donne un JSON avec les clés {param_keys}.\n\n"
        f"{param_descriptions}\n"
        f"Forme attendue: {json_structure}.\n"
        "Fiche technique:\n'''\n"
        f"{pdf_text}\n'''")

    messages = [
        {
            "role": "system",
            "content": "You extract information from technical datasheets and answer only with JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]

    response = call_vllm_with_messages(messages, temperature=0.0)
    content = response["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}

    # Nettoyage des valeurs pour chaque paramètre
    result = {}
    for param_name in FICHES_TECHNIQUES_PARAMS.keys():
        value = str(data.get(param_name, "")).strip()
        
        if value.lower() in {"inconnu", "unknown", "n/a", "non spécifié"}:
            value = ""
        
        # Vérification spécifique pour le fournisseur
        if param_name == "fournisseur" and value and _is_excluded_name(value):
            value = ""
        
        result[param_name] = value

    return result


def _text_is_usable(text: str) -> bool:
    """Détermine si le texte extrait est exploitable."""
    from config import MIN_TEXT_LENGTH
    return len(text.strip()) >= MIN_TEXT_LENGTH


def display_field(value: Any) -> str:
    """Affiche un champ avec icône si vide/inconnu."""
    if value is None or value == "" or str(value).strip().lower() in {"inconnu", "unknown", "n/a"}:
        return "❌"
    return str(value).strip()


def render_fiches_techniques_table(
    results: List[Dict[str, Any]],
    total_seconds: float,
) -> None:
    """Affiche les résultats des fiches techniques en ligne."""
    
    if not results:
        return

    # Titre avec provider et modèle
    if LLM_PROVIDER == "gemini":
        model_display = GEMINI_MODEL
    elif LLM_PROVIDER == "deepinfra":
        model_display = DEEPINFRA_MODEL
    else:
        model_display = MODEL_NAME
    
    print(f"\nRésultats avec le provider {LLM_PROVIDER.upper()} et le modèle {model_display} :", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    # Calculer la longueur maximale des labels pour aligner les deux-points
    labels = [param.replace("_", " ").title() for param in FICHES_TECHNIQUES_PARAMS.keys()]
    max_label_length = max(len(label) for label in labels)
    
    for entry in results:
        extraction = entry.get("extraction", {}) or {}
        filename = entry.get("file", "")
        # Retirer l'extension .pdf pour un affichage plus compact
        if filename.endswith(".pdf"):
            filename = filename[:-4]
        
        # En-tête du fichier
        print(f"\n** {filename} {'*' * (78 - len(filename) - 4)}", file=sys.stderr)
        
        # Affichage de chaque paramètre avec alignement
        for param_name, label in zip(FICHES_TECHNIQUES_PARAMS.keys(), labels):
            value = display_field(extraction.get(param_name))
            # Aligner les deux-points
            print(f"-  {label.ljust(max_label_length)} : {value}", file=sys.stderr)

    # Statistiques finales
    print("\n" + "=" * 80, file=sys.stderr)
    count = len(results)
    avg = total_seconds / count if count else 0.0
    print(
        f"Durée totale: {total_seconds:.2f} s | Fiches: {count} | Moyenne: {avg:.2f} s",
        file=sys.stderr,
    )


def run_fiches_techniques() -> None:
    """Boucle principale d'analyse de fiches techniques."""
    
    fiches_dir = Path(FICHES_TECHNIQUES_DIR).expanduser().resolve()

    provider_info = f"Provider LLM : {LLM_PROVIDER.upper()}"
    if LLM_PROVIDER == "gemini":
        provider_info += f" (modèle : {GEMINI_MODEL})"
    elif LLM_PROVIDER == "deepinfra":
        provider_info += f" (modèle : {DEEPINFRA_MODEL})"
    else:
        provider_info += f" (modèle : {MODEL_NAME})"
    print(provider_info, file=sys.stderr)

    try:
        pdf_files = list_pdf_files(fiches_dir)
    except FileNotFoundError as err:
        raise SystemExit(str(err)) from err

    results = []
    total_start = time.perf_counter()

    for pdf_path in pdf_files:
        file_start = time.perf_counter()
        print(f"Processing {pdf_path.name}…", file=sys.stderr)
        pdf_text = extract_text(pdf_path)

        # Déterminer le mode : texte ou vision
        use_vision = not _text_is_usable(pdf_text)

        if use_vision:
            print(f"  Mode vision activé (texte insuffisant)", file=sys.stderr)
            images_b64 = pdf_to_base64_images(pdf_path)
            prompt = build_prompt("")  # prompt sans texte PDF
            api_response = call_vllm_vision(prompt, images_b64, temperature=0.0)
        else:
            prompt = build_prompt(pdf_text)
            api_response = call_vllm(prompt, temperature=0.0)

        content = api_response["choices"][0]["message"]["content"].strip()
        
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            parsed_json = parse_llm_json(content)

        # Nettoyage des valeurs pour chaque paramètre
        cleaned_values = {}
        missing_params = []
        
        for param_name in FICHES_TECHNIQUES_PARAMS.keys():
            value = str(parsed_json.get(param_name, "")).strip()
            
            if value.lower() in {"inconnu", "unknown", "n/a", "non spécifié"}:
                value = ""
            
            # Vérification spécifique pour le fournisseur
            if param_name == "fournisseur" and value and _is_excluded_name(value):
                value = ""
            
            cleaned_values[param_name] = value
            
            if not value:
                missing_params.append(param_name)

        # Requête ciblée si informations manquantes
        if missing_params:
            details = ask_fiche_technique_details(pdf_text)
            for param_name in missing_params:
                if details.get(param_name):
                    cleaned_values[param_name] = details[param_name]

        parsed_json = cleaned_values

        duration = time.perf_counter() - file_start

        results.append({
            "file": pdf_path.name,
            "extraction": parsed_json,
            "duration_s": duration,
        })

    total_duration = time.perf_counter() - total_start
    render_fiches_techniques_table(results, total_duration)


if __name__ == "__main__":  # pragma: no cover
    run_fiches_techniques()
