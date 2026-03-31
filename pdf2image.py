#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour extraire les images d'un PDF
Usage: python pdf2image.py <chemin_du_pdf>
"""

import sys
import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io

# ========================================
# PARAMÈTRES DE CONFIGURATION
# ========================================
# Nombre de pixels à rogner en haut de chaque image
CROP_TOP = 145

# Nombre de pixels à rogner en bas de chaque image
CROP_BOTTOM = 30
# ========================================


def extract_images_from_pdf(pdf_path):
    """
    Extrait toutes les images d'un PDF et les sauvegarde dans un dossier
    
    Args:
        pdf_path: Chemin vers le fichier PDF
    """
    # Vérifier que le fichier existe
    if not os.path.exists(pdf_path):
        print(f"Erreur : Le fichier '{pdf_path}' n'existe pas")
        sys.exit(1)
    
    # Vérifier que c'est bien un PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Erreur : Le fichier '{pdf_path}' n'est pas un PDF")
        sys.exit(1)
    
    # Créer le dossier de sortie avec le même nom que le PDF
    pdf_name = Path(pdf_path).stem
    output_dir = os.path.join(os.path.dirname(pdf_path), pdf_name)
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dossier de sortie : {output_dir}")
    
    # Ouvrir le PDF
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Erreur lors de l'ouverture du PDF : {e}")
        sys.exit(1)
    
    image_counter = 10  # Commence à 10, incrémente de 10 en 10
    total_images = 0
    
    # Parcourir chaque page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        # Extraire chaque image de la page
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                # Extraire l'image
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Créer le nom du fichier avec le préfixe numéroté
                image_filename = f"{image_counter:04d}_image.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                # Rogner l'image si nécessaire
                if CROP_TOP > 0 or CROP_BOTTOM > 0:
                    try:
                        # Charger l'image avec PIL
                        img = Image.open(io.BytesIO(image_bytes))
                        width, height = img.size
                        
                        # Calculer les nouvelles dimensions
                        top = CROP_TOP
                        bottom = height - CROP_BOTTOM
                        
                        # Vérifier que les dimensions sont valides
                        if bottom > top and top >= 0:
                            # Rogner l'image (left, top, right, bottom)
                            img_cropped = img.crop((0, top, width, bottom))
                            
                            # Sauvegarder l'image rognée
                            img_cropped.save(image_path, quality=95, optimize=False)
                            print(f"Page {page_num + 1:2d} → {image_filename} (rogné {CROP_TOP}px haut, {CROP_BOTTOM}px bas)")
                        else:
                            # Si les dimensions sont invalides, sauvegarder sans rogner
                            with open(image_path, "wb") as image_file:
                                image_file.write(image_bytes)
                            print(f"Page {page_num + 1:2d} → {image_filename} (rognage ignoré: dimensions invalides)")
                    except Exception as crop_error:
                        # En cas d'erreur de rognage, sauvegarder l'image originale
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_bytes)
                        print(f"Page {page_num + 1:2d} → {image_filename} (erreur rognage: {crop_error})")
                else:
                    # Pas de rognage, sauvegarder directement
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    print(f"Page {page_num + 1:2d} → {image_filename}")
                
                image_counter += 10
                total_images += 1
                
            except Exception as e:
                print(f"Page {page_num + 1} → Erreur lors de l'extraction de l'image {img_index + 1} : {e}")
    
    # Fermer le PDF
    pdf_document.close()
    
    print(f"\n✓ Extraction terminée : {total_images} image(s) extraite(s)")
    print(f"✓ Images sauvegardées dans : {output_dir}")


def main():
    """Point d'entrée du script"""
    if len(sys.argv) != 2:
        print("Usage: python pdf2image.py <chemin_du_pdf>")
        print("Exemple: python pdf2image.py mon_document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extract_images_from_pdf(pdf_path)


if __name__ == "__main__":
    main()
