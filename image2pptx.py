#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour créer une présentation PPTX à partir des images d'un dossier
avec logos, titre et numérotation des pages
Usage: python image2pptx.py <chemin_du_dossier>
"""

import sys
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from PIL import Image

# ========================================
# PARAMÈTRES DE CONFIGURATION
# ========================================
# Dimensions de la diapositive (format paysage A4)
SLIDE_WIDTH_CM = 28.0
SLIDE_HEIGHT_CM = 21.0

# Zone de titre en haut
TITLE_HEIGHT_CM = 2.0  # Hauteur de la barre de titre
TITLE_MARGIN_CM = 0.5  # Marge au-dessus et en dessous du titre

# Logos (laisser vide si pas de logo)
LOGO_LEFT = "/home/tony/Documents/InfoSaone/Clients/Plastigray/Projets/2026/16-Présentation IA/infosaone-0600px.png"   # Chemin vers le logo à gauche du titre
LOGO_RIGHT = "/home/tony/Documents/InfoSaone/Clients/Plastigray/Projets/2026/16-Présentation IA/LogoPGEuronyl-300px.png"  # Chemin vers le logo à droite du titre
LOGO_HEIGHT_CM = 1.5  # Hauteur des logos

# Taille de la police pour le numéro de page
PAGE_NUMBER_SIZE = Pt(10)

# Marges autour de l'image
IMAGE_MARGIN_CM = 0.5

# Titres des diapositives (un titre par diapositive)
# Si la liste est vide, le nom du dossier sera utilisé pour toutes les diapos
# Si la liste contient moins de titres que d'images, le nom du dossier sera utilisé pour les diapos restantes
SLIDE_TITLES = [
    "Présentation IA pour CODIR Plastigray",
    "IA de type Chat GPT les plus populaires à ce jour utilisables en SAAS",

    "Qui possède quoi",
    "ChatGPT - Gemini",
    "Copilot - Claude",
    "Perplexity - Qwen",


    "C'est quoi un LLM Open Source",
    "Présentation de Qwen (Alibaba) (Open Source)",
    "Achat d'un serveur IA DELL GB10",
    "VLLM à quoi ça sert",
    "Installation de Open WebUI",
    "Exemples de prompts dans Open WebUI",
    "Exemple de programmes pour exploiter l'API de VLMM",
    #"Intégration avec Odoo",
    "Recherche générale - Listes",
    "Recherche générale - Tableaux croisés",
    "Recherche générale - Graphiques",
    "Fiche technique par IA",
    #"Ce qu'il serait possible de faire aussi",
]
# ========================================





def get_image_dimensions(image_path):
    """
    Récupère les dimensions d'une image
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        Tuple (largeur, hauteur) en pixels
    """
    with Image.open(image_path) as img:
        return img.size


def add_header_elements(slide, slide_width, slide_height, slide_title, page_num, total_pages):
    """
    Ajoute les éléments d'en-tête : logos, titre et numéro de page
    
    Args:
        slide: La diapositive
        slide_width: Largeur de la diapositive
        slide_height: Hauteur de la diapositive
        slide_title: Titre de la diapositive
        page_num: Numéro de la page actuelle
        total_pages: Nombre total de pages
    """
    # Calcul des largeurs (15% - 70% - 15%)
    logo_width = slide_width * 0.15
    title_width = slide_width * 0.70
    
    # Marges latérales pour le titre (espace entre logos et titre)
    title_side_margin = Cm(0.6)
    
    # Position Y du titre et des logos (pas de marge en haut)
    title_y = 0
    
    # Logo gauche (si défini)
    if LOGO_LEFT and os.path.exists(LOGO_LEFT):
        logo_height = Cm(LOGO_HEIGHT_CM)
        # Calculer la largeur en gardant les proportions
        img_width, img_height = get_image_dimensions(LOGO_LEFT)
        logo_aspect = img_width / img_height
        logo_calc_width = logo_height * logo_aspect
        
        # Positionner à gauche
        slide.shapes.add_picture(
            LOGO_LEFT,
            left=0,
            top=title_y,
            height=logo_height
        )
    
    # Logo droite (si défini)
    if LOGO_RIGHT and os.path.exists(LOGO_RIGHT):
        logo_height = Cm(LOGO_HEIGHT_CM)
        # Calculer la largeur en gardant les proportions
        img_width, img_height = get_image_dimensions(LOGO_RIGHT)
        logo_aspect = img_width / img_height
        logo_calc_width = logo_height * logo_aspect
        
        # Positionner à droite
        slide.shapes.add_picture(
            LOGO_RIGHT,
            left=int(slide_width - logo_calc_width),
            top=title_y,
            height=logo_height
        )
    
    # Zone de titre au centre - couvre toute la largeur de la page
    title_box = slide.shapes.add_textbox(
        left=0,
        top=title_y,
        width=int(slide_width),
        height=Cm(TITLE_HEIGHT_CM)
    )
    title_frame = title_box.text_frame
    title_frame.text = slide_title
    title_frame.word_wrap = True  # Retour à la ligne automatique
    title_frame.vertical_anchor = 1  # Alignement vertical au milieu (MSO_ANCHOR.MIDDLE)
    # Marges internes pour éviter les logos
    title_frame.margin_left = int(logo_width + title_side_margin)
    title_frame.margin_right = int(logo_width + title_side_margin)
    # Centrer le texte horizontalement
    para = title_frame.paragraphs[0]
    para.font.size = Pt(18)
    para.font.bold = True
    para.font.language_id = 0x040C  # Français (France)
    para.alignment = PP_ALIGN.CENTER
    
    # Numéro de page en bas à droite (pas de marge à droite ni en dessous)
    page_num_box = slide.shapes.add_textbox(
        left=int(slide_width - Cm(2.5)),
        top=int(slide_height - Cm(0.8)),
        width=Cm(2.5),
        height=Cm(0.8)
    )
    page_frame = page_num_box.text_frame
    page_frame.text = f"{page_num}/{total_pages}"
    page_frame.paragraphs[0].font.size = PAGE_NUMBER_SIZE
    page_frame.paragraphs[0].font.language_id = 0x040C  # Français (France)
    page_frame.paragraphs[0].alignment = 2  # Droite (PP_ALIGN.RIGHT = 2)


def calculate_image_position_and_size(img_width, img_height, slide_width, slide_height):
    """
    Calcule la position et la taille optimale pour centrer l'image
    en tenant compte de la zone de titre
    
    Args:
        img_width: Largeur de l'image en pixels
        img_height: Hauteur de l'image en pixels
        slide_width: Largeur de la diapositive
        slide_height: Hauteur de la diapositive
    
    Returns:
        Tuple (left, top, width, height) pour positionner l'image
    """
    # Zone disponible pour l'image (sous le titre)
    title_zone_height = Cm(TITLE_HEIGHT_CM + 2 * TITLE_MARGIN_CM)
    margin = Cm(IMAGE_MARGIN_CM)
    
    available_width = slide_width - (2 * margin)
    available_height = slide_height - title_zone_height - (2 * margin)
    
    # Ratio de l'image
    img_ratio = img_width / img_height
    available_ratio = available_width / available_height
    
    # Calculer les dimensions en conservant les proportions
    if img_ratio > available_ratio:
        # L'image est plus large : on limite par la largeur
        width = available_width
        height = width / img_ratio
    else:
        # L'image est plus haute : on limite par la hauteur
        height = available_height
        width = height * img_ratio
    
    # Centrer l'image horizontalement
    left = (slide_width - width) / 2
    # Positionner l'image verticalement (sous le titre, centrée dans l'espace restant)
    top = title_zone_height + ((slide_height - title_zone_height - height) / 2)
    
    return left, top, width, height



def images_to_pptx(folder_path):
    """
    Crée une présentation PPTX avec une diapositive par image
    
    Args:
        folder_path: Chemin vers le dossier contenant les images
    """
    # Vérifier que le dossier existe
    if not os.path.exists(folder_path):
        print(f"Erreur : Le dossier '{folder_path}' n'existe pas")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Erreur : '{folder_path}' n'est pas un dossier")
        sys.exit(1)
    
    # Extensions d'images supportées
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'}
    
    # Récupérer toutes les images du dossier
    image_files = []
    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    if not image_files:
        print(f"Erreur : Aucune image trouvée dans le dossier '{folder_path}'")
        sys.exit(1)
    
    print(f"Trouvé {len(image_files)} image(s) à ajouter à la présentation")
    
    # Créer la présentation
    print("Création d'une nouvelle présentation PPTX")
    prs = Presentation()
    
    # Définir les dimensions de la diapositive (format paysage A4)
    prs.slide_width = Cm(SLIDE_WIDTH_CM)
    prs.slide_height = Cm(SLIDE_HEIGHT_CM)
    
    slide_width = prs.slide_width
    slide_height = prs.slide_height
    
    # Layout vierge (blank)
    blank_layout = prs.slide_layouts[6]  # Layout 6 est généralement vide
    
    # Nom du dossier pour le titre
    folder_name = os.path.basename(os.path.normpath(folder_path))
    total_pages = len(image_files)
    
    # Ajouter une diapositive pour chaque image
    for idx, image_path in enumerate(image_files, 1):
        try:
            # Créer une nouvelle diapositive vierge
            slide = prs.slides.add_slide(blank_layout)
            
            # Définir le fond blanc
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(255, 255, 255)
            
            # Déterminer le titre de cette diapositive
            if SLIDE_TITLES and idx <= len(SLIDE_TITLES):
                slide_title = SLIDE_TITLES[idx - 1]
            else:
                slide_title = folder_name
            
            # Ajouter les éléments d'en-tête (logos, titre, numéro de page)
            add_header_elements(slide, slide_width, slide_height, slide_title, idx, total_pages)
            
            # Obtenir les dimensions de l'image
            img_width, img_height = get_image_dimensions(image_path)
            
            # Calculer la position et la taille optimale
            left, top, width, height = calculate_image_position_and_size(
                img_width, img_height,
                slide_width, slide_height
            )
            
            # Ajouter l'image à la diapositive
            slide.shapes.add_picture(
                image_path,
                left=int(left),
                top=int(top),
                width=int(width),
                height=int(height)
            )
            
            print(f"Diapo {idx:2d}/{total_pages} → {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Erreur lors de l'ajout de {image_path} : {e}")
    
    # Sauvegarder la présentation
    pptx_path = os.path.join(folder_path, f"{folder_name}.pptx")
    
    try:
        prs.save(pptx_path)
        print(f"\n✓ Présentation PPTX créée : {pptx_path}")
        print(f"✓ Nombre de diapositives : {len(image_files)}")
        
        # Afficher la taille du fichier
        file_size = os.path.getsize(pptx_path)
        if file_size < 1024 * 1024:
            print(f"✓ Taille du fichier : {file_size / 1024:.1f} Ko")
        else:
            print(f"✓ Taille du fichier : {file_size / (1024 * 1024):.1f} Mo")
        
        # Informations sur les logos
        if LOGO_LEFT or LOGO_RIGHT:
            print("\nLogos configurés :")
            if LOGO_LEFT:
                if os.path.exists(LOGO_LEFT):
                    print(f"  ✓ Logo gauche : {LOGO_LEFT}")
                else:
                    print(f"  ⚠ Logo gauche non trouvé : {LOGO_LEFT}")
            if LOGO_RIGHT:
                if os.path.exists(LOGO_RIGHT):
                    print(f"  ✓ Logo droite : {LOGO_RIGHT}")
                else:
                    print(f"  ⚠ Logo droite non trouvé : {LOGO_RIGHT}")
        else:
            print("\nℹ Aucun logo configuré. Pour ajouter des logos, modifiez :")
            print("  LOGO_LEFT = '/chemin/vers/logo_gauche.png'")
            print("  LOGO_RIGHT = '/chemin/vers/logo_droite.png'")
            
    except Exception as e:
        print(f"Erreur lors de la création de la présentation : {e}")
        sys.exit(1)



def main():
    """Point d'entrée du script"""
    if len(sys.argv) != 2:
        print("Usage: python image2pptx.py <chemin_du_dossier>")
        print("Exemple: python image2pptx.py ./mes_images")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    images_to_pptx(folder_path)


if __name__ == "__main__":
    main()
