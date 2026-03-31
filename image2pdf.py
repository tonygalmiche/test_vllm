#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour créer un PDF à partir des images d'un dossier
Usage: python image2pdf.py <chemin_du_dossier>
"""

import sys
import os
from pathlib import Path
from PIL import Image


def images_to_pdf(folder_path):
    """
    Crée un PDF avec une page par image à partir des images d'un dossier
    
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
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
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
    
    print(f"Trouvé {len(image_files)} image(s) à convertir")
    
    # Nom du PDF : nom du dossier
    folder_name = os.path.basename(os.path.normpath(folder_path))
    pdf_path = os.path.join(folder_path, f"{folder_name}.pdf")
    
    # Convertir les images en RGB si nécessaire et les charger
    images = []
    for idx, image_file in enumerate(image_files, 1):
        try:
            img = Image.open(image_file)
            
            # Convertir en RGB si nécessaire (pour RGBA, P, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Créer un fond blanc
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
            print(f"Image {idx:2d} → {os.path.basename(image_file)}")
            
        except Exception as e:
            print(f"Erreur lors du chargement de {image_file} : {e}")
    
    if not images:
        print("Erreur : Aucune image n'a pu être chargée")
        sys.exit(1)
    
    # Sauvegarder toutes les images dans un PDF
    try:
        # La première image est utilisée pour créer le PDF, les autres sont ajoutées
        images[0].save(
            pdf_path,
            save_all=True,
            append_images=images[1:] if len(images) > 1 else [],
            resolution=100.0,
            quality=95,
            optimize=False
        )
        
        print(f"\n✓ PDF créé avec succès : {pdf_path}")
        print(f"✓ Nombre de pages : {len(images)}")
        
        # Afficher la taille du fichier
        file_size = os.path.getsize(pdf_path)
        if file_size < 1024 * 1024:
            print(f"✓ Taille du fichier : {file_size / 1024:.1f} Ko")
        else:
            print(f"✓ Taille du fichier : {file_size / (1024 * 1024):.1f} Mo")
            
    except Exception as e:
        print(f"Erreur lors de la création du PDF : {e}")
        sys.exit(1)


def main():
    """Point d'entrée du script"""
    if len(sys.argv) != 2:
        print("Usage: python image2pdf.py <chemin_du_dossier>")
        print("Exemple: python image2pdf.py ./mes_images")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    images_to_pdf(folder_path)


if __name__ == "__main__":
    main()
