#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour créer une présentation ODP à partir des images d'un dossier
Usage: python image2odp.py <chemin_du_dossier>
"""

import sys
import os
from pathlib import Path
from odf.opendocument import OpenDocumentPresentation
from odf.style import Style, MasterPage, PageLayout, PageLayoutProperties, DrawingPageProperties, ParagraphProperties, TextProperties
from odf.draw import Page, Frame, Image as DrawImage, TextBox
from odf.text import P
from PIL import Image

# ========================================
# PARAMÈTRES DE CONFIGURATION
# ========================================
# Dimensions de la diapositive (en cm)
SLIDE_WIDTH_CM = 28.0   # Largeur standard (28cm = ~11 inches)
SLIDE_HEIGHT_CM = 21.0  # Hauteur standard (21cm = ~8.3 inches)

# Marges autour de l'image (en cm)
MARGIN_CM = 1.0

# Zone de titre
TITLE_HEIGHT_CM = 2.0   # Hauteur de la zone de titre
TITLE_MARGIN_CM = 0.5   # Marge du titre

# Logos (laisser vide "" si pas de logo)
LOGO_LEFT = "/home/tony/Documents/InfoSaone/Clients/Plastigray/Projets/2026/16-Présentation IA/infosaone-0600px.png"   # Chemin vers le logo à gauche du titre
LOGO_RIGHT = "/home/tony/Documents/InfoSaone/Clients/Plastigray/Projets/2026/16-Présentation IA/LogoPGEuronyl-300px.png"  # Chemin vers le logo à droite du titre
LOGO_HEIGHT_CM = 1.5  # Hauteur des logos

# Numérotation des pages
PAGE_NUMBER_SIZE = "10pt"  # Taille de la police pour la numérotation
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


def calculate_position_and_size(img_width, img_height, slide_width_cm, slide_height_cm, margin_cm):
    """
    Calcule la position et la taille optimale pour centrer l'image
    tout en conservant ses proportions
    
    Args:
        img_width: Largeur de l'image en pixels
        img_height: Hauteur de l'image en pixels
        slide_width_cm: Largeur de la diapositive en cm
        slide_height_cm: Hauteur de la diapositive en cm
        margin_cm: Marge à respecter en cm
    
    Returns:
        Tuple (x, y, width, height) en cm pour positionner l'image
    """
    # Zone disponible pour l'image (en tenant compte de la zone de titre)
    available_width = slide_width_cm - (2 * margin_cm)
    available_height = slide_height_cm - (2 * margin_cm) - TITLE_HEIGHT_CM - TITLE_MARGIN_CM
    
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
    
    # Centrer l'image horizontalement et la positionner sous le titre
    x = (slide_width_cm - width) / 2
    y = TITLE_HEIGHT_CM + TITLE_MARGIN_CM + margin_cm + (available_height - height) / 2
    
    return x, y, width, height


def setup_presentation_styles(doc):
    """
    Configure les styles de base pour la présentation
    
    Args:
        doc: Document OpenDocumentPresentation
    
    Returns:
        Le nom du master page à utiliser
    """
    # Style de mise en page
    pl = PageLayout(name="PageLayout1")
    pl.addElement(PageLayoutProperties(
        pagewidth=f"{SLIDE_WIDTH_CM}cm",
        pageheight=f"{SLIDE_HEIGHT_CM}cm"
    ))
    doc.automaticstyles.addElement(pl)
    
    # Style de drawing page avec fond blanc
    dpstyle = Style(name="dp1", family="drawing-page")
    dpstyle.addElement(DrawingPageProperties(
        backgroundsize="full",
        fillcolor="#ffffff",
        fill="solid"
    ))
    doc.automaticstyles.addElement(dpstyle)
    
    # Style de paragraphe pour le titre
    title_para_style = Style(name="TitleParagraph", family="paragraph")
    title_para_style.addElement(ParagraphProperties(
        textalign="center"
    ))
    title_para_style.addElement(TextProperties(
        fontsize="24pt",
        fontweight="bold"
    ))
    doc.automaticstyles.addElement(title_para_style)
    
    # Style de paragraphe pour la numérotation de page
    page_num_style = Style(name="PageNumber", family="paragraph")
    page_num_style.addElement(ParagraphProperties(
        textalign="end"  # Aligné à droite
    ))
    page_num_style.addElement(TextProperties(
        fontsize=PAGE_NUMBER_SIZE
    ))
    doc.automaticstyles.addElement(page_num_style)
    
    # Master page
    mp = MasterPage(name="Standard", pagelayoutname=pl)
    
    # Dimensions fixes pour les logos et le titre (en pourcentage de la largeur)
    logo_width_cm = SLIDE_WIDTH_CM * 0.15  # 15% de la largeur pour chaque logo
    title_width_cm = SLIDE_WIDTH_CM * 0.70  # 70% de la largeur pour le titre
    
    # Ajouter les logos au master page s'ils sont définis
    if LOGO_LEFT and os.path.exists(LOGO_LEFT):
        logo_left_frame = Frame(
            width=f"{logo_width_cm}cm",
            height=f"{TITLE_HEIGHT_CM}cm",
            x="0cm",
            y=f"{TITLE_MARGIN_CM}cm"
        )
        href_left = doc.addPicture(LOGO_LEFT)
        logo_left_img = DrawImage(href=href_left)
        logo_left_frame.addElement(logo_left_img)
        mp.addElement(logo_left_frame)
    
    if LOGO_RIGHT and os.path.exists(LOGO_RIGHT):
        logo_right_frame = Frame(
            width=f"{logo_width_cm}cm",
            height=f"{TITLE_HEIGHT_CM}cm",
            x=f"{SLIDE_WIDTH_CM - logo_width_cm}cm",
            y=f"{TITLE_MARGIN_CM}cm"
        )
        href_right = doc.addPicture(LOGO_RIGHT)
        logo_right_img = DrawImage(href=href_right)
        logo_right_frame.addElement(logo_right_img)
        mp.addElement(logo_right_frame)
    
    # Position du titre centré (15% + 70% + 15% = 100%)
    title_x_cm = logo_width_cm
    
    # Ajouter le cadre de titre dans le master page
    master_title_frame = Frame(
        width=f"{title_width_cm}cm",
        height=f"{TITLE_HEIGHT_CM}cm",
        x=f"{title_x_cm}cm",
        y=f"{TITLE_MARGIN_CM}cm"
    )
    master_title_textbox = TextBox()
    master_title_para = P(stylename="TitleParagraph", text="Cliquez pour éditer le titre")
    master_title_textbox.addElement(master_title_para)
    master_title_frame.addElement(master_title_textbox)
    mp.addElement(master_title_frame)
    
    # Ajouter le cadre de numérotation de page dans le master page
    page_num_width = 2.0
    page_num_height = 0.8
    page_num_margin = 0.3
    
    master_pagenum_frame = Frame(
        width=f"{page_num_width}cm",
        height=f"{page_num_height}cm",
        x=f"{SLIDE_WIDTH_CM - page_num_width - page_num_margin}cm",
        y=f"{SLIDE_HEIGHT_CM - page_num_height - page_num_margin}cm"
    )
    master_pagenum_textbox = TextBox()
    master_pagenum_para = P(stylename="PageNumber", text="1/1")
    master_pagenum_textbox.addElement(master_pagenum_para)
    master_pagenum_frame.addElement(master_pagenum_textbox)
    mp.addElement(master_pagenum_frame)
    
    doc.masterstyles.addElement(mp)
    
    return "Standard", title_x_cm, title_width_cm


def images_to_odp(folder_path):
    """
    Crée une présentation ODP avec une diapositive par image
    
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
    doc = OpenDocumentPresentation()
    
    # Configurer les styles et obtenir les dimensions du titre
    masterpagename, title_x, title_width = setup_presentation_styles(doc)
    
    # Ajouter une diapositive pour chaque image
    total_slides = len(image_files)
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            # Créer une nouvelle page (diapositive)
            page = Page(stylename="dp1", masterpagename=masterpagename)
            
            # Ajouter un cadre de titre par dessus celui du master page (pour pouvoir l'éditer par page)
            title_frame = Frame(
                width=f"{title_width}cm",
                height=f"{TITLE_HEIGHT_CM}cm",
                x=f"{title_x}cm",
                y=f"{TITLE_MARGIN_CM}cm"
            )
            
            # Ajouter une boîte de texte dans le cadre
            textbox = TextBox()
            title_para = P(stylename="TitleParagraph", text="Cliquez pour éditer le titre")
            textbox.addElement(title_para)
            title_frame.addElement(textbox)
            page.addElement(title_frame)
            
            # Obtenir les dimensions de l'image
            img_width, img_height = get_image_dimensions(image_path)
            
            # Calculer la position et la taille optimale
            x, y, width, height = calculate_position_and_size(
                img_width, img_height,
                SLIDE_WIDTH_CM, SLIDE_HEIGHT_CM,
                MARGIN_CM
            )
            
            # Ajouter l'image au document et obtenir le href
            href = doc.addPicture(image_path)
            
            # Créer un cadre pour l'image
            frame = Frame(
                width=f"{width}cm",
                height=f"{height}cm",
                x=f"{x}cm",
                y=f"{y}cm"
            )
            
            # Créer l'élément image et l'ajouter au cadre
            image_element = DrawImage(href=href)
            frame.addElement(image_element)
            
            # Ajouter le cadre à la page
            page.addElement(frame)
            
            # Ajouter la numérotation de page en bas à droite (par dessus celle du master page)
            page_num_width = 2.0  # Largeur de la zone de numérotation
            page_num_height = 0.8  # Hauteur de la zone de numérotation
            page_num_margin = 0.3  # Marge depuis le bord
            
            page_num_frame = Frame(
                width=f"{page_num_width}cm",
                height=f"{page_num_height}cm",
                x=f"{SLIDE_WIDTH_CM - page_num_width - page_num_margin}cm",
                y=f"{SLIDE_HEIGHT_CM - page_num_height - page_num_margin}cm"
            )
            
            page_num_textbox = TextBox()
            page_num_para = P(stylename="PageNumber", text=f"{idx}/{total_slides}")
            page_num_textbox.addElement(page_num_para)
            page_num_frame.addElement(page_num_textbox)
            page.addElement(page_num_frame)
            
            # Ajouter la page au document
            doc.presentation.addElement(page)
            
            print(f"Diapo {idx:2d} → {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Erreur lors de l'ajout de {image_path} : {e}")
    
    # Sauvegarder la présentation
    folder_name = os.path.basename(os.path.normpath(folder_path))
    odp_path = os.path.join(folder_path, f"{folder_name}.odp")
    
    try:
        doc.save(odp_path)
        
        print(f"\n✓ Présentation ODP créée : {odp_path}")
        print(f"✓ Nombre de diapositives : {len(image_files)}")
        
        # Afficher la taille du fichier
        file_size = os.path.getsize(odp_path)
        if file_size < 1024 * 1024:
            print(f"✓ Taille du fichier : {file_size / 1024:.1f} Ko")
        else:
            print(f"✓ Taille du fichier : {file_size / (1024 * 1024):.1f} Mo")
            
    except Exception as e:
        print(f"Erreur lors de la création de la présentation : {e}")
        sys.exit(1)


def main():
    """Point d'entrée du script"""
    if len(sys.argv) != 2:
        print("Usage: python image2odp.py <chemin_du_dossier>")
        print("Exemple: python image2odp.py ./mes_images")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    images_to_odp(folder_path)


if __name__ == "__main__":
    main()
