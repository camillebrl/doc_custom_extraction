import os
import json
import shutil
import jsonlines
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from pdf2image import convert_from_path
import cv2
from PIL import Image as PILImage
import numpy as np
from werkzeug.utils import secure_filename
import easyocr
from docling.document_converter import DocumentConverter
import pytesseract
import Levenshtein
import uuid
import random
import albumentations as alb


app = Flask(__name__)

# Initialize EasyOCR reader
# Using English by default, you can add more languages as needed
reader = easyocr.Reader(['en'])

# Temporary storage directories
TEMP_DIR = Path("temp_images")
TEMP_DIR.mkdir(exist_ok=True)
MARKED_DIR = Path("temp_marked")
MARKED_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Dictionary to store OCR results
ocr_results = {}
# Path for storing jsonl annotations
JSONL_ANNOT_FILE = Path("temp_annot.jsonl")

# Initialiser les annotations en mémoire avec un dictionnaire vide
annotations = {} 

class SafeRotation(alb.ImageOnlyTransform):
    """
    Applique une rotation légère à l'image sans déformer le texte excessivement.
    Limité à de petits angles pour préserver la lisibilité du texte.
    """
    def __init__(self, limit=5, border_mode=cv2.BORDER_CONSTANT, 
                 value=(255, 255, 255), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.limit = limit
        self.border_mode = border_mode
        self.value = value
        
    def apply(self, img, **params):
        angle = random.uniform(-self.limit, self.limit)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculer la matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Appliquer la rotation
        rotated = cv2.warpAffine(
            img, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.value
        )
        
        return rotated
    
class Erosion(alb.ImageOnlyTransform):
    """
    Apply erosion operation to an image.
    """
    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(alb.ImageOnlyTransform):
    """
    Apply dilation operation to an image.
    """
    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(alb.ImageOnlyTransform):
    """
    Apply a bitmap-style transformation to an image.
    """
    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


class RandomPaperTexture(alb.ImageOnlyTransform):
    """
    Ajoute une texture de papier en arrière-plan.
    Cette transformation ne modifie pas la position des objets dans l'image.
    """
    def __init__(self, textures_strength=(0.02, 0.15), always_apply=False, p=0.3):
        super().__init__(always_apply=always_apply, p=p)
        self.textures_strength = textures_strength
        
    def apply(self, img, **params):
        # Créer une texture de papier
        texture = np.ones_like(img) * 240  # Légèrement grisâtre
        noise = np.random.randint(0, 15, img.shape).astype(np.uint8)
        texture = np.clip(texture - noise, 0, 255).astype(np.uint8)
        
        # Mélanger la texture avec l'image
        alpha = random.uniform(self.textures_strength[0], self.textures_strength[1])
        # Préserver le texte (zones sombres)
        mask = (img < 200).astype(np.float32)
        result = img.copy().astype(np.float32)
        
        # Appliquer la texture uniquement aux zones blanches
        result = mask * img + (1 - mask) * ((1 - alpha) * img + alpha * texture)
        
        return np.clip(result, 0, 255).astype(np.uint8)


class RandomInkSpot(alb.ImageOnlyTransform):
    """
    Simule des taches d'encre aléatoires sans déplacer les éléments existants.
    """
    def __init__(self, num_spots=(1, 3), size_range=(1, 5), intensity=(20, 60), always_apply=False, p=0.15):
        super().__init__(always_apply=always_apply, p=p)
        self.num_spots = num_spots
        self.size_range = size_range
        self.intensity = intensity
        
    def apply(self, img, **params):
        result = img.copy()
        h, w = img.shape[:2]
        num = random.randint(self.num_spots[0], self.num_spots[1])
        
        for _ in range(num):
            # Position aléatoire
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            
            # Taille aléatoire
            size = random.randint(self.size_range[0], self.size_range[1])
            
            # Intensité aléatoire (plus petit = plus sombre)
            intensity = random.randint(self.intensity[0], self.intensity[1])
            
            # Créer la tache
            cv2.circle(result, (x, y), size, (intensity, intensity, intensity), -1)
            
        return result


class RandomNoiseInText(alb.ImageOnlyTransform):
    """
    Ajoute du bruit principalement dans les zones de texte.
    """
    def __init__(self, noise_range=(5, 30), always_apply=False, p=0.2):
        super().__init__(always_apply=always_apply, p=p)
        self.noise_range = noise_range
        
    def apply(self, img, **params):
        result = img.copy()
        
        # Créer un masque pour les zones de texte (zones sombres)
        text_mask = (img < 150)
        
        # Générer du bruit aléatoire
        noise_amount = random.randint(self.noise_range[0], self.noise_range[1])
        noise = np.random.randint(-noise_amount, noise_amount, img.shape, dtype=np.int16)
        
        # Appliquer le bruit principalement aux zones de texte
        text_noise = noise * 0.8 * text_mask
        bg_noise = noise * 0.2 * (~text_mask)
        total_noise = text_noise + bg_noise
        
        # Appliquer le bruit à l'image
        result = np.clip(result.astype(np.int16) + total_noise, 0, 255).astype(np.uint8)
        
        return result


class RandomPageShadow(alb.ImageOnlyTransform):
    """
    Ajoute une ombre de page, comme si la page était légèrement courbée lors de la numérisation.
    Ne modifie pas la position des éléments.
    """
    def __init__(self, shadow_width_range=(0.1, 0.3), intensity_range=(10, 40), always_apply=False, p=0.25):
        super().__init__(always_apply=always_apply, p=p)
        self.shadow_width_range = shadow_width_range
        self.intensity_range = intensity_range
        
    def apply(self, img, **params):
        h, w = img.shape[:2]
        result = img.copy()
        
        # Décider du côté pour l'ombre (gauche, droite)
        side = random.choice(['left', 'right'])
        
        # Calculer la largeur de l'ombre
        shadow_width = int(w * random.uniform(self.shadow_width_range[0], self.shadow_width_range[1]))
        
        # Créer un masque de gradient pour l'ombre
        if side == 'left':
            x = np.arange(shadow_width)
            gradient = np.tile(x, (h, 1)) / shadow_width
            shadow_mask = np.zeros((h, w))
            shadow_mask[:, :shadow_width] = gradient
        else:  # right
            x = np.arange(shadow_width)[::-1]
            gradient = np.tile(x, (h, 1)) / shadow_width
            shadow_mask = np.zeros((h, w))
            shadow_mask[:, -shadow_width:] = gradient
        
        # Calculer l'intensité de l'ombre
        intensity = random.randint(self.intensity_range[0], self.intensity_range[1])
        
        # Appliquer l'ombre à l'image
        shadow = shadow_mask[:, :, np.newaxis] * intensity
        result = np.clip(result.astype(np.int16) - shadow.astype(np.int16), 0, 255).astype(np.uint8)
        
        return result

def enhance_image_for_ocr(image_path):
    """
    Améliore l'image pour faciliter l'OCR en ajoutant du contraste et en réduisant le flou
    """
    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convertir en niveaux de gris si l'image est en couleur
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1. Amélioration du contraste par égalisation d'histogramme adaptative
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    # # 3. Réduction du flou par filtre de netteté
    # kernel = np.array([[-1, -1, -1], 
    #                    [-1,  9, -1], 
    #                    [-1, -1, -1]])
    # sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 4. Binarisation adaptative pour améliorer la détection du texte
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Créer un nom pour l'image améliorée
    enhanced_path = str(TEMP_DIR / f"{os.path.basename(image_path)}")
    
    # Sauvegarder l'image améliorée
    cv2.imwrite(enhanced_path, binary)
    
    return enhanced_path

def is_new_bbox(new_bbox, existing_bboxes, overlap_threshold=0.5):
    """
    Vérifie si une bounding box est nouvelle (pas suffisamment de chevauchement avec les existantes)
    
    Args:
        new_bbox: Nouvelle bbox à vérifier [x1, y1, x2, y2]
        existing_bboxes: Liste des bboxes existantes [[x1, y1, x2, y2], ...]
        overlap_threshold: Seuil de chevauchement (0-1)
        
    Returns:
        True si la bbox est nouvelle, False sinon
    """
    # Extraire les coordonnées
    x1_new, y1_new, x2_new, y2_new = new_bbox
    
    # Calculer l'aire de la nouvelle bbox
    area_new = (x2_new - x1_new) * (y2_new - y1_new)
    if area_new <= 0:
        return False  # Bbox invalide
    
    for bbox in existing_bboxes:
        x1, y1, x2, y2 = bbox
        
        # Calculer l'intersection
        x_left = max(x1_new, x1)
        y_top = max(y1_new, y1)
        x_right = min(x2_new, x2)
        y_bottom = min(y2_new, y2)
        
        if x_right <= x_left or y_bottom <= y_top:
            continue  # Pas de chevauchement avec cette bbox
        
        # Calculer l'aire d'intersection
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculer le ratio de chevauchement par rapport à la nouvelle bbox
        overlap_ratio = intersection / area_new
        
        if overlap_ratio > overlap_threshold:
            return False  # Chevauchement significatif, donc pas nouvelle
    
    return True  # Aucun chevauchement significatif trouvé, donc nouvelle

def clear_annotation_files():
    """Clear annotation files on startup"""    
    # Vider le fichier JSONL
    with open(JSONL_ANNOT_FILE, 'w') as f:
        f.write('')
    
    print("Cleared annotation files on startup")




def get_existing_files():
    if not TEMP_DIR.exists():
        return []
    
    files = []
    for file in TEMP_DIR.glob("*"):
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            files.append(str(file))
    return files

def save_document_only(file):
    """
    Save uploaded file to temp_images.
    Convert PDFs to PNG pages.
    Do NOT process OCR.
    Return list of image filepaths.
    """
    if not file:
        return []
        
    saved = []
    filename = secure_filename(file.filename)
    dst = TEMP_DIR / filename
    file.save(dst)
    
    if dst.suffix.lower() == ".pdf":
        pages = convert_from_path(dst)
        for i, page in enumerate(pages, start=1):
            img_name = f"{dst.stem}_page{i}.png"
            img_path = TEMP_DIR / img_name
            page.save(img_path, "PNG")
            saved.append(str(img_path))
    else:
        saved.append(str(dst))
    
    return saved


def combine_with_no_chevauchement(image_path, combined_data, data2, source1, source2):
    """
    Combines OCR data from two sources while avoiding overlapping bounding boxes.
    
    Args:
        image_path: Path to the image being processed
        combined_data: Dictionary containing the primary OCR data (words, bboxes, etc.)
        data2: Dictionary containing the secondary OCR data to be added
        source1: Name of the primary OCR engine
        source2: Name of the secondary OCR engine
        
    Returns:
        The combined data dictionary with non-overlapping results from both sources
    """
    # Ensure combined_data is a dictionary with required keys
    if not isinstance(combined_data, dict):
        print(f"Error: combined_data must be a dictionary, got {type(combined_data)}")
        return combined_data
        
    # Add results from source2 that don't overlap with those already in combined_data
    for i, (word, bbox) in enumerate(zip(data2["words"], data2["bboxes"])):
        if is_new_bbox(bbox, combined_data["bboxes"], overlap_threshold=0.3):
            if len(word.strip()) > 1:  # Only add non-empty words
                combined_data["words"].append(word)
                combined_data["bboxes"].append(bbox)
                combined_data["page_numbers"].append(1)
                combined_data["engine"].append(source2)
    
    # Store the combined results
    if combined_data["words"]:
        print(f"Found {len(combined_data['words'])} words in {image_path} "
              f"({len([e for e in combined_data['engine'] if e == source1])} from {source1}, "
              f"{len([e for e in combined_data['engine'] if e == source2])} from {source2})")
    else:
        print(f"No text detected in {image_path}")
    
    return combined_data

def process_ocr(image_path):
    """
    Traite l'OCR avec EASYOCR puis complète avec Tesseract
    """
    try:
        # 1. Utiliser EasyOCR en premier
        easyocr_data = extract_text_with_easyocr(image_path)
        
        # 2. Utiliser Tesseract ensuite
        tesseract_data = extract_text_with_tesseract(image_path)

        # 3. Utiliser Docling ensuite
        docling_data = extract_text_with_docling(image_path)
        
        # 4. Combine les résultats
        combined_data = {
            "words": easyocr_data["words"].copy(),
            "bboxes": easyocr_data["bboxes"].copy(),
            "page_numbers": [1] * len(easyocr_data["words"]),
            "engine": ["easyocr"] * len(easyocr_data["words"])
        }
        combined_data = combine_with_no_chevauchement(image_path, combined_data, tesseract_data, "easyocr", "tesseract")
        combined_data = combine_with_no_chevauchement(image_path, combined_data, docling_data, "easyocr+tesseract", "docling")

        # 5. Stocker les résultats combinés
        ocr_results[image_path] = combined_data

        return True, f"Found {len(combined_data['words'])} words in {image_path}"
    
    except Exception as e:
        print(f"No text detected in {image_path}")
        return False, "No text detected in image"

    
    except Exception as e:
        print(f"Error processing OCR for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing OCR: {str(e)}"

def apply_augmentation(image_path, transform_idx, aug_id):
    """
    Applique une augmentation à une image et sauvegarde le résultat
    
    Args:
        image_path: Chemin de l'image originale
        transform_idx: Index de la transformation à appliquer
        aug_id: Identifiant unique pour cette augmentation
        
    Returns:
        Chemin de l'image augmentée ou None en cas d'erreur
    """
    try:
        # Lire l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur: Impossible de lire l'image {image_path}")
            return None
        
        # Pour cette version simplifiée, on applique juste quelques transformations basiques
        # plutôt que d'utiliser les transformations albumentations
        
        # Liste de transformations simples de base
        transformations = [
            # 0. Augmenter le contraste
            lambda image: cv2.convertScaleAbs(image, alpha=1.3, beta=10),
            
            # 1. Ajouter du bruit gaussien
            lambda image: cv2.add(image, 
                                 np.random.normal(0, 15, image.shape).astype(np.uint8)),
            
            # 2. Rotation légère
            lambda image: cv2.warpAffine(
                image, 
                cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 
                                        random.uniform(-3, 3), 1.0),
                (image.shape[1], image.shape[0])
            ),
            
            # 3. Flou léger
            lambda image: cv2.GaussianBlur(image, (3, 3), 0),
            
            # 4. Distorsion de perspective
            lambda image: apply_perspective_transform(image),
            
            # 5. Ajustement de luminosité
            lambda image: cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), 
                                            beta=random.uniform(-10, 10))
        ]
        
        # Choisir la transformation en fonction de l'index
        transform = transformations[transform_idx % len(transformations)]
        
        # Appliquer la transformation
        augmented = transform(img)
        
        # Créer un nom pour l'image augmentée
        path = Path(image_path)
        aug_filename = f"{path.stem}_aug{aug_id}{path.suffix}"
        aug_path = TEMP_DIR / aug_filename
        
        # Sauvegarder l'image augmentée
        cv2.imwrite(str(aug_path), augmented)
        print(f"Image augmentée sauvegardée: {aug_path}")
        
        return str(aug_path)
    
    except Exception as e:
        print(f"Erreur lors de l'augmentation de l'image: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_perspective_transform(image):
    h, w = image.shape[:2]
    
    # Définir les points source (les coins de l'image)
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    # Définir les points destination (légèrement déplacés)
    dst_points = np.float32([
        [np.random.randint(0, 30), np.random.randint(0, 30)],
        [w - 1 - np.random.randint(0, 30), np.random.randint(0, 30)],
        [np.random.randint(0, 30), h - 1 - np.random.randint(0, 30)],
        [w - 1 - np.random.randint(0, 30), h - 1 - np.random.randint(0, 30)]
    ])
    
    # Calculer la matrice de transformation
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Appliquer la transformation
    transformed = cv2.warpPerspective(image, M, (w, h))
    
    return transformed

def generate_augmentations(image_path, n_augmentations=5):
    """
    Génère des versions augmentées d'une image et réapplique l'OCR
    
    Args:
        image_path: Chemin de l'image originale
        n_augmentations: Nombre d'augmentations à générer
        
    Returns:
        Liste des chemins des images augmentées
    """
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"L'image {image_path} n'existe pas.")
        return []
    
    # Vérifier si nous avons des annotations pour cette image
    if image_path not in annotations or "text_regions" not in annotations[image_path]:
        print(f"Aucune annotation trouvée pour {image_path}")
        return []
    
    # Récupérer les annotations originales
    original_annotations = annotations[image_path]["text_regions"]
    if not original_annotations:
        print(f"Aucune annotation trouvée pour {image_path}")
        return []
    
    augmented_paths = []
    
    # Générer les augmentations
    for i in range(n_augmentations):
        # Générer un ID unique pour cette augmentation
        aug_id = str(i+1)
        
        # Appliquer une augmentation aléatoire
        aug_path = apply_augmentation(image_path, i, aug_id)
        if not aug_path:
            continue
            
        augmented_paths.append(aug_path)
        
        # Traiter l'OCR pour l'image augmentée
        success, _ = process_ocr(aug_path)
        if not success or aug_path not in ocr_results:
            print(f"Échec de l'OCR pour {aug_path}")
            continue
        
        # Initialiser les annotations pour l'image augmentée
        if aug_path not in annotations:
            # Récupérer les dimensions de l'image augmentée
            img = cv2.imread(aug_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            annotations[aug_path] = {
                "dimensions": {"width": width, "height": height},
                "text_regions": []
            }
        
        # Transférer les labels des annotations originales vers les nouveaux mots OCR
        transfer_annotations(image_path, aug_path)
        
        # Sauvegarder les annotations au format JSONL
        save_to_jsonl(aug_path)
    
    return augmented_paths

def transfer_annotations(original_path, augmented_path):
    """
    Transfère les annotations de l'image originale vers l'image augmentée en utilisant
    la correspondance par distance de Levenshtein entre les textes OCR
    """
    if original_path not in ocr_results or augmented_path not in ocr_results:
        return False
    
    original_ocr = ocr_results[original_path]
    augmented_ocr = ocr_results[augmented_path]
    
    # Récupérer les annotations originales
    original_annotations = annotations[original_path]["text_regions"]
    
    # Dictionnaire pour stocker les labels par texte
    text_to_label = {}
    
    # Construire une correspondance texte -> label à partir des annotations originales
    for ann in original_annotations:
        text = ann["text"]
        label = ann.get("label", "O")
        text_to_label[text] = label
    
    # Seuil de similarité accepté pour la correspondance (plus petit = plus strict)
    SIMILARITY_THRESHOLD = 0.2  # 20% de différence maximale tolérée
    
    # Correspondance par similarité de texte
    for i, (aug_word, aug_bbox) in enumerate(zip(augmented_ocr["words"], augmented_ocr["bboxes"])):
        best_match = None
        best_ratio = 0
        
        # Chercher le mot le plus similaire dans les mots originaux annotés
        for orig_text, label in text_to_label.items():
            # Calculer la similarité avec la distance de Levenshtein normalisée
            length = max(len(aug_word), len(orig_text))
            if length == 0:  # Éviter la division par zéro
                continue
                
            # Calculer la distance de Levenshtein
            distance = Levenshtein.distance(aug_word.lower(), orig_text.lower())
            
            # Convertir la distance en ratio de similarité (1 = identique, 0 = complètement différent)
            similarity_ratio = 1 - (distance / length)
            
            # Garder le meilleur match
            if similarity_ratio > best_ratio and similarity_ratio > (1 - SIMILARITY_THRESHOLD):
                best_ratio = similarity_ratio
                best_match = (orig_text, label)
        
        # Si un match suffisamment bon a été trouvé, ajouter l'annotation
        if best_match:
            orig_text, label = best_match
            
            # Ajouter une annotation pour ce mot dans l'image augmentée
            annotations[augmented_path]["text_regions"].append({
                "bbox": aug_bbox,
                "text": aug_word,  # Utiliser le texte OCR de l'image augmentée
                "label": label,    # Mais le label du texte original
                "match_confidence": best_ratio  # Stocker le niveau de confiance pour référence
            })
            
            print(f"Match trouvé: '{aug_word}' -> '{orig_text}' (confiance: {best_ratio:.2f}, label: {label})")
    
    return True

def draw_annotations(image_path, current_bbox=None):
    if not image_path or not os.path.exists(image_path):
        return None
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Store dimensions if not already stored
    if image_path in annotations:
        if "dimensions" not in annotations[image_path]:
            annotations[image_path]["dimensions"] = {"width": width, "height": height}
    else:
        annotations[image_path] = {
            "dimensions": {"width": width, "height": height},
            "text_regions": []
        }
    
    # Draw OCR detected text boxes (light gray)
    if image_path in ocr_results:
        for bbox in ocr_results[image_path]["bboxes"]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (200, 200, 200), 1)
    
    # Group annotations by their base label (removing B-/I- prefixes)
    # This will help visualize multi-box entities with the same color
    grouped_annotations = {}
    
    if image_path in annotations and "text_regions" in annotations[image_path]:
        for ann in annotations[image_path]["text_regions"]:
            label = ann.get("label", "O")
            base_label = label
            
            # Strip B-/I- prefixes to get the base label
            if label.startswith("B-") or label.startswith("I-"):
                base_label = label[2:]
                
            # Initialize group if it doesn't exist
            if base_label not in grouped_annotations:
                grouped_annotations[base_label] = []
                
            # Add annotation to its group
            grouped_annotations[base_label].append(ann)
    
    # Define a fixed set of colors for different entity types
    # This will ensure that all boxes with the same base label have the same color
    color_map = {
        "O": (255, 255, 0),      # Yellow for "O" (outside) labels
        "ADDRESS": (255, 0, 0),   # Red
        "COMPANY": (0, 255, 0),   # Green
        "DATE": (0, 0, 255),      # Blue
        "EMAIL": (255, 0, 255),   # Magenta
        "NAME": (0, 255, 255),    # Cyan
        "PHONE": (255, 128, 0),   # Orange
        "TOTAL": (128, 0, 255),   # Purple
        "OTHER": (128, 128, 128)  # Gray
    }
    
    # Draw existing annotations by group
    for base_label, anns in grouped_annotations.items():
        # Choose color based on base label
        color = color_map.get(base_label, (255, 0, 0))  # Default to red
        
        for ann in anns:
            bbox = ann["bbox"]
            text = ann["text"]
            label = ann.get("label", "O")
            
            # Draw semi-transparent highlight for text
            overlay = img_rgb.copy()
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
            cv2.addWeighted(overlay, 0.3, img_rgb, 0.7, 0, img_rgb)
            
            # Draw border around text
            cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw text and label on top
            display_text = text
            if label and label != "O":
                display_text = f"{text} [{label}]"
            cv2.putText(img_rgb, display_text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw current selection
    if current_bbox:
        x1, y1, x2, y2 = current_bbox
        overlay = img_rgb.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_rgb, 0.7, 0, img_rgb)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Save marked image to static directory for display
    output_filename = f"marked_{Path(image_path).name}"
    output_path = STATIC_DIR / output_filename
    
    # Convert back to BGR for saving
    cv2.imwrite(str(output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    return output_filename

def add_annotation(image_path, text_content, x1, y1, x2, y2, label="O", save_to_file=False):
    """
    Add text annotation with coordinates and label
    If save_to_file is True, also save to JSON and JSONL files
    """        
    if not image_path or not text_content:
        return False, "No image or text specified."
        
    try:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        bbox = [x1, y1, x2, y2]
        
        # Initialize image annotations if not exist
        if image_path not in annotations:
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image."
            height, width = img.shape[:2]
            
            annotations[image_path] = {
                "dimensions": {"width": width, "height": height},
                "text_regions": []
            }
        elif "text_regions" not in annotations[image_path]:
            annotations[image_path]["text_regions"] = []
        
        # Add new text region
        annotations[image_path]["text_regions"].append({
            "bbox": bbox,
            "text": text_content,
            "label": label
        })
        
        # Save to file ONLY if explicitly requested
        if save_to_file:
            # Save to JSONL
            save_to_jsonl(image_path)
        
        # Draw annotations on image
        draw_annotations(image_path)
        
        return True, f"Saved annotation for {Path(image_path).name}."
    except Exception as e:
        return False, f"Error saving annotation: {str(e)}"

def save_to_jsonl(image_path=None):
    """
    Sauvegarde les annotations au format JSONL attendu pour l'entraînement
    Si image_path est None, sauvegarder toutes les images
    """
    # Si aucun chemin d'image n'est fourni, traiter toutes les images
    if image_path is None:
        
        # Traiter chaque image annotée
        for img_path in annotations.keys():
            save_single_image_to_jsonl(img_path)
        
        return True, f"Saved annotations for {len(annotations)} images to {JSONL_ANNOT_FILE}"
    else:
        # Traiter uniquement l'image spécifiée
        if image_path not in annotations or "text_regions" not in annotations[image_path]:
            return False, f"No annotations found for {image_path}"
        
        save_single_image_to_jsonl(image_path)
        return True, f"Saved annotations for {image_path} to {JSONL_ANNOT_FILE}"

def save_single_image_to_jsonl(image_path):
    """
    Sauvegarde les annotations d'une seule image au format JSONL
    avec prise en charge des tags BIO
    """
    if image_path not in annotations or "text_regions" not in annotations[image_path]:
        return
    
    # Récupérer les dimensions de l'image
    width = annotations[image_path]["dimensions"]["width"]
    height = annotations[image_path]["dimensions"]["height"]
    image_size = (width, height)
    
    # Préparer les données
    words = []
    bboxes = []
    labels = []
    
    # Si nous avons des résultats OCR pour cette image
    if image_path in ocr_results:
        # Utiliser tous les mots de l'OCR comme base
        words = ocr_results[image_path]["words"]
        bboxes = ocr_results[image_path]["bboxes"]
        # Par défaut, tous les mots sont étiquetés "O" (Outside)
        labels = ["O"] * len(words)
        
        # Mettre à jour les labels pour les mots annotés
        for region in annotations[image_path]["text_regions"]:
            region_text = region["text"]
            region_bbox = region["bbox"]
            region_label = region.get("label", "O")
            
            # Trouver le mot correspondant dans les résultats OCR
            for i, (word, bbox) in enumerate(zip(words, bboxes)):
                # Vérifier si le mot correspond (exact match ou contenu dans le texte région)
                if word == region_text or (word in region_text and is_bbox_overlap(bbox, region_bbox)):
                    labels[i] = region_label
    else:
        # Si pas d'OCR, utiliser uniquement les régions annotées
        for region in annotations[image_path]["text_regions"]:
            words.append(region["text"])
            bboxes.append(region["bbox"])
            labels.append(region.get("label", "O"))
    # Normaliser les bounding boxes
    norm_boxes = normalize_bboxes(bboxes, image_size)
    
    # Créer le dictionnaire à sauvegarder
    jsonl_data = {
        "image_path": image_path,
        "words": words,
        "bboxes": norm_boxes,
        "labels": labels
    }
    
    # Sauvegarder au format JSONL (append)
    with jsonlines.open(JSONL_ANNOT_FILE, mode='a') as writer:
        writer.write(jsonl_data)

def is_bbox_overlap(bbox1, bbox2, threshold=0.5):
    """
    Vérifie si deux bounding boxes se chevauchent avec un seuil donné
    en tenant compte des deux perspectives (boîte 1 et boîte 2)
    """
    # Extraire les coordonnées
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculer la surface de l'intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return False  # Pas de chevauchement
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculer les surfaces des deux bboxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Éviter la division par zéro
    if area1 == 0 or area2 == 0:
        return False
    
    # Calculer les deux ratios de chevauchement
    ratio1 = intersection / area1
    ratio2 = intersection / area2
    
    # Utiliser le maximum des deux ratios (plus permissif)
    # On pourrait aussi utiliser min(ratio1, ratio2) pour être plus strict
    overlap_ratio = max(ratio1, ratio2)
    
    return overlap_ratio >= threshold

def get_image_dims(image_path):
    if not image_path or not os.path.exists(image_path):
        return 0, 0
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return w, h  # Return width, height
    except:
        return 0, 0
    
def extract_text_with_tesseract(image_path):
    """
    Extraire le texte et les bounding boxes d'une image en utilisant Tesseract OCR
    """
    try:
        # Charger l'image avec PIL
        img = PILImage.open(image_path)
        image_height = img.height  # Obtenir la hauteur de l'image pour l'inversion des coordonnées
        
        # Utiliser Tesseract pour extraire le texte et les données
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        extracted_data = {
            "words": [],
            "bboxes": [],
            "confidences": []
        }
        
        # Parcourir les mots détectés
        for i in range(len(data['text'])):
            # Ignorer les entrées vides
            if not data['text'][i].strip():
                continue
                
            # Récupérer les coordonnées
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            # Créer la bbox au format x1,y1,x2,y2 avec les coordonnées Y inversées
            # pour correspondre au système de coordonnées de Docling
            # Pour Docling, (0,0) est en bas à gauche, pour Tesseract, (0,0) est en haut à gauche
            bbox_tuple = (x, image_height - (y + h), x + w, image_height - y)
            
            # Ajouter à nos données
            extracted_data["words"].append(data['text'][i])
            extracted_data["bboxes"].append(bbox_tuple)
            extracted_data["confidences"].append(data['conf'][i])
        
        return extracted_data
    
    except Exception as e:
        print(f"Erreur lors de l'extraction avec Tesseract: {e}")
        import traceback
        traceback.print_exc()
        return {"words": [], "bboxes": [], "confidences": []}

def extract_text_with_easyocr(image_path):
    """
    Extraire le texte et les bounding boxes d'une image en utilisant EasyOCR
    """
    try:
        # Améliorer l'image avant OCR
        enhanced_image_path = enhance_image_for_ocr(image_path)
        if enhanced_image_path is None:
            return {"words": [], "bboxes": [], "confidences": []}
        
        # Charger l'image avec PIL pour obtenir les dimensions
        img = PILImage.open(enhanced_image_path)
        image_height = img.height  # Obtenir la hauteur de l'image pour l'inversion des coordonnées
        
        # Utiliser EasyOCR pour extraire le texte
        results = reader.readtext(enhanced_image_path)
        
        extracted_data = {
            "words": [],
            "bboxes": [],
            "confidences": []
        }
        
        # Parcourir les résultats détectés
        for result in results:
            bbox = result[0]  # EasyOCR retourne 4 points (polygone)
            text = result[1]   # Le texte reconnu
            confidence = result[2]  # Score de confiance
            
            if not text.strip():
                continue
            
            # Convertir les 4 points du polygone en rectangle (x1,y1,x2,y2)
            # Prendre les coordonnées min et max pour obtenir le rectangle englobant
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            
            # Créer la bbox au format x1,y1,x2,y2 avec les coordonnées Y inversées
            # pour correspondre au système de coordonnées attendu (0,0 en bas à gauche)
            bbox_tuple = (int(x1), image_height - int(y2), int(x2), image_height - int(y1))
            
            # Ajouter à nos données
            extracted_data["words"].append(text)
            extracted_data["bboxes"].append(bbox_tuple)
            extracted_data["confidences"].append(confidence)
        
        return extracted_data
    
    except Exception as e:
        print(f"Erreur lors de l'extraction avec EasyOCR: {e}")
        import traceback
        traceback.print_exc()
        return {"words": [], "bboxes": [], "confidences": []}

def extract_text_with_docling(image_path):
    """
    Extraire le texte et les bounding boxes d'une image en utilisant Docling
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(image_path)
        doc = result.document
        
        extracted_data = {
            "words": [],
            "bboxes": [],
            "page_numbers": []
        }
        
        for text_item in doc.texts:
            for prov in text_item.prov:
                # Récupérer le texte
                text = text_item.text.strip()
                if not text:
                    continue
                
                # Récupérer la bounding box
                bbox = prov.bbox
                bbox_tuple = bbox.as_tuple()  # x1, y1, x2, y2 format
                
                # Ajouter à nos données
                extracted_data["words"].append(text)
                extracted_data["bboxes"].append(bbox_tuple)
                extracted_data["page_numbers"].append(prov.page_no)
        
        return extracted_data
    
    except Exception as e:
        print(f"Erreur lors de l'extraction avec Docling: {e}")
        import traceback
        traceback.print_exc()
        return {"words": [], "bboxes": [], "page_numbers": []}

def normalize_bboxes(bboxes, image_size, scale=1000):
    """
    Normaliser les bounding boxes à l'échelle 0-1000 (format attendu par LayoutLMv3)
    
    Args:
        bboxes: Liste de bounding boxes au format [x1, y1, x2, y2]
        image_size: Tuple (width, height) de l'image
        scale: Facteur d'échelle (1000 par défaut pour LayoutLMv3)
    
    Returns:
        Liste de bounding boxes normalisées à l'échelle 0-1000
    """
    width, height = image_size
    normalized_bboxes = []
    
    for box in bboxes:
        # Normaliser à l'échelle 0-1000
        normalized_box = [
            int(box[0] / width * scale),
            int(box[1] / height * scale),
            int(box[2] / width * scale),
            int(box[3] / height * scale)
        ]
        normalized_bboxes.append(normalized_box)
    
    return normalized_bboxes

def export_to_docling(output_path="docling_annotations.json"):
    """
    Export annotations to a format compatible with docling
    """
    docling_data = []
    
    for image_path, ann_data in annotations.items():
        if "text_regions" not in ann_data or not ann_data["text_regions"]:
            continue
        
        image_name = os.path.basename(image_path)
        
        # Extract dimensions
        width = ann_data["dimensions"]["width"]
        height = ann_data["dimensions"]["height"]
        
        # Create docling item
        docling_item = {
            "image": image_name,
            "width": width,
            "height": height,
            "text_regions": []
        }
        
        # Add text regions
        for region in ann_data["text_regions"]:
            x1, y1, x2, y2 = region["bbox"]
            text = region["text"]
            label = region.get("label", "O")
            
            region_data = {
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "label": label
            }
            
            docling_item["text_regions"].append(region_data)
        
        docling_data.append(docling_item)

def clean_jsonl_file(file_path):
    """
    Nettoie le fichier JSONL en ne gardant que la dernière occurrence de chaque image_path
    Retourne le nombre d'entrées après nettoyage
    """
    if not os.path.exists(file_path):
        return 0
    
    # Lire toutes les entrées du fichier
    entries = []
    with jsonlines.open(file_path, mode='r') as reader:
        for line in reader:
            entries.append(line)
    
    # Garder seulement la dernière entrée pour chaque image_path
    unique_entries = {}
    for entry in entries:
        image_path = entry.get('image_path')
        if image_path:
            unique_entries[image_path] = entry
    
    # Écrire les entrées uniques dans le fichier
    with jsonlines.open(file_path, mode='w') as writer:
        for entry in unique_entries.values():
            writer.write(entry)
    
    return len(unique_entries)





# Flask routes
@app.route('/')
def index():
    files = get_existing_files()
    return render_template('index.html', files=files)

@app.route('/upload_only', methods=['POST'])
def upload_only():
    """Upload file without OCR processing"""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        })
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
        
    try:
        saved = save_document_only(file)
        
        if not saved:
            return jsonify({
                'status': 'error',
                'message': 'Failed to save document'
            })
            
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'files': get_existing_files()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}'
        })

@app.route('/get_files', methods=['GET'])
def get_files():
    """Get list of all available files"""
    files = get_existing_files()
    return jsonify(files)

@app.route('/image/<path:filename>')
def image(filename):
    return send_from_directory(TEMP_DIR, os.path.basename(filename))

@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route('/process_ocr/<path:filename>')
def run_ocr(filename):
    """Run OCR processing on an image"""
    if os.path.exists(filename):
        success, message = process_ocr(filename)
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'data': ocr_results.get(filename, {})
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            })
    
    return jsonify({
        'status': 'error',
        'message': 'File not found'
    })

@app.route('/annotate', methods=['POST'])
def annotate():
    data = request.json
    image_path = data.get('image_path')
    text = data.get('text')
    x1 = int(data.get('x1', 0))
    y1 = int(data.get('y1', 0))
    x2 = int(data.get('x2', 0))
    y2 = int(data.get('y2', 0))
    label = data.get('label', 'O')
    save_to_file = data.get('save_to_file', False)
    
    # Validation for B-/I- label format
    if label != 'O' and not (label.startswith('B-') or label.startswith('I-')):
        # If the label doesn't already have a B- or I- prefix, assume it's an error
        # and add B- by default
        label = 'B-' + label
    
    # Ajouter l'annotation à l'image originale
    success, message = add_annotation(image_path, text, x1, y1, x2, y2, label, save_to_file)
    
    if success:
        marked_file = draw_annotations(image_path)
        current_annotations = annotations.get(image_path, {}).get("text_regions", [])
        
        # Ne PAS générer les augmentations ici
        
        return jsonify({
            'status': 'success',
            'message': message,
            'marked_file': marked_file,
            'annotations': current_annotations
        })
    else:
        return jsonify({
            'status': 'error',
            'message': message
        })

@app.route('/export_annotations', methods=['POST'])
def export_annotations():
    """Export all annotations to JSONL file"""
    try:
        success, message = save_to_jsonl(None)  # Exporter toutes les images
        if success:
            return jsonify({
                'status': 'success',
                'message': message
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error exporting annotations: {str(e)}'
        })

@app.route('/export_single_annotation', methods=['POST'])
def export_single_annotation():
    """Export annotations for a single image to JSONL file"""
    try:
        data = request.json
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'No image path provided'
            })
            
        if image_path not in annotations:
            return jsonify({
                'status': 'error',
                'message': f'No annotations found for {image_path}'
            })
            
        # Sauvegarder uniquement les annotations de l'image spécifiée dans JSONL
        success, message = save_to_jsonl(image_path)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Annotations for {Path(image_path).name} saved to file.'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error exporting annotation: {str(e)}'
        })

@app.route('/preview', methods=['POST'])
def preview():
    data = request.json
    image_path = data.get('image_path')
    x1 = int(data.get('x1', 0))
    y1 = int(data.get('y1', 0))
    x2 = int(data.get('x2', 0))
    y2 = int(data.get('y2', 0))
    
    marked_file = draw_annotations(image_path, [x1, y1, x2, y2])
    return jsonify({'marked_file': marked_file})

@app.route('/dimensions/<path:filename>')
def dimensions(filename):
    width, height = get_image_dims(filename)
    return jsonify({'width': width, 'height': height})

@app.route('/annotations/<path:filename>')
def get_annotations(filename):
    if filename in annotations and "text_regions" in annotations[filename]:
        return jsonify(annotations[filename]["text_regions"])
    return jsonify([])

@app.route('/ocr_results/<path:filename>')
def get_ocr_results(filename):
    if filename in ocr_results:
        return jsonify(ocr_results[filename])
    else:
        # Si les résultats OCR ne sont pas encore disponibles, essayons de traiter l'OCR maintenant
        if os.path.exists(filename):
            success, _ = process_ocr(filename)
            if success and filename in ocr_results:
                return jsonify(ocr_results[filename])
    
    return jsonify({"words": [], "bboxes": [], "page_numbers": []})

@app.route('/get_labels', methods=['GET'])
def get_labels():
    """Get all unique labels from annotations, removing B-/I- prefixes"""
    unique_base_labels = set(["O"])  # Always include "O" (Outside) label
    
    for image_path, ann_data in annotations.items():
        if "text_regions" in ann_data:
            for region in ann_data["text_regions"]:
                if "label" in region and region["label"]:
                    label = region["label"]
                    
                    # Remove B- or I- prefix if present
                    if label.startswith("B-") or label.startswith("I-"):
                        base_label = label[2:]
                        unique_base_labels.add(base_label)
                    else:
                        unique_base_labels.add(label)
    
    # Convert set to list for JSON serialization
    return jsonify(list(unique_base_labels))

@app.route('/add_label', methods=['POST'])
def add_label():
    """Add a new label to be used in annotations"""
    data = request.json
    new_label = data.get('label')
    
    if not new_label or not isinstance(new_label, str):
        return jsonify({
            'status': 'error',
            'message': 'Invalid label'
        })
    
    # We don't need to store labels separately as they're attached to annotations
    # Just return success
    return jsonify({
        'status': 'success',
        'message': f'Added label: {new_label}'
    })

@app.route('/reset_annotations', methods=['POST'])
def reset_annotations():
    """Reset all annotations in memory"""
    global annotations
    annotations = {}
    
    # Vider également les fichiers
    clear_annotation_files()
    
    return jsonify({
        'status': 'success',
        'message': 'All annotations have been reset'
    })

@app.route('/check_ocr_status/<path:filename>')
def check_ocr_status(filename):
    """Vérifie si l'OCR a déjà été traité pour cette image sans relancer le traitement"""
    if filename in ocr_results and ocr_results[filename]["words"]:
        return jsonify({
            'status': 'success',
            'message': 'OCR results already available',
            'has_ocr': True,
            'word_count': len(ocr_results[filename]["words"])
        })
    else:
        return jsonify({
            'status': 'info',
            'message': 'No OCR results found for this image',
            'has_ocr': False
        })

@app.after_request
def add_header(response):
    # Désactiver le cache pour toutes les réponses
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/finish', methods=['POST'])
def finish_annotation():
    """
    Nettoie le fichier JSONL en supprimant les doublons et arrête l'application
    """
    try:
        # Nettoyer le fichier JSONL
        count = clean_jsonl_file(JSONL_ANNOT_FILE)
        
        # Préparer la réponse
        response = {
            'status': 'success',
            'message': f'Cleaned annotation file: {count} unique entries saved.',
            'shutdown': True
        }
        
        # Planifier l'arrêt de l'application
        def shutdown_server():
            import time
            time.sleep(2)  # Attendre 2 secondes pour que la réponse soit envoyée
            os._exit(0)    # Arrêter l'application
            
        import threading
        threading.Thread(target=shutdown_server).start()
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during finish: {str(e)}',
            'shutdown': False
        })
    
@app.route('/generate_augmentations', methods=['POST'])
def generate_augments():
    """
    Route dédiée à la génération d'augmentations après sauvegarde dans le fichier JSONL
    """
    try:
        data = request.json
        image_path = data.get('image_path')
        n_augmentations = data.get('n_augmentations', 5)
        
        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'No image path provided'
            })
            
        if image_path not in annotations:
            return jsonify({
                'status': 'error',
                'message': f'No annotations found for {image_path}'
            })
        
        # Génère les augmentations
        augmented_paths = generate_augmentations(image_path, n_augmentations=n_augmentations)
        
        return jsonify({
            'status': 'success',
            'message': f'Generated {len(augmented_paths)} augmented images',
            'augmented_images': augmented_paths
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating augmentations: {str(e)}'
        })

if __name__ == "__main__":
    
    # Ensure directories exist
    TEMP_DIR.mkdir(exist_ok=True)
    MARKED_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Copy index.html to templates directory if necessary
    index_html_path = templates_dir / "index.html"
    
    print("Starting Text Annotation Tool on http://127.0.0.1:5001")
    app.run(debug=True, host="127.0.0.1", port=5001)