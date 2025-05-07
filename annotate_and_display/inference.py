import base64
import io
import json
import os
from typing import Any, Dict, List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
from flask import Flask, jsonify, render_template, request
from matplotlib.patches import Rectangle
from PIL import Image as PILImage
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

matplotlib.use('Agg') 

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = '../layoutlmv3_ft/results/final_model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialisation du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger les mappings de labels
with open(os.path.join(MODEL_PATH, "label_mappings.json"), "r") as f:
    label_maps = json.load(f)

id2label = {int(k): v for k, v in label_maps["id2label"].items()}
label2id = label_maps["label2id"]

print(f"Chargement du modèle depuis {MODEL_PATH}...")
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # Mettre le modèle en mode évaluation
print("Modèle chargé avec succès!")

# Mapping des couleurs pour les labels
COLOR_MAP = {
    "PATIENT": "yellow",
    "DOCTOR": "blue",
    "FACILITY": "purple",
    "ACT": "green",
    "DOC": "orange"
}

def convert_nested_numpy_types(obj):
    """Recursively convert numpy types in nested data structures to Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_nested_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nested_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_nested_numpy_types(item) for item in obj)
    return obj

def detect_image_blocks(image, gradient_thresh=6000, min_size=10, smooth=1, recursive=False, depth=0, max_depth=2):
    """Détecte les blocs d'image en utilisant l'analyse de gradient
    et retourne les coordonnées des blocs plutôt que des sous-images.
    
    Args:
        image: Image d'entrée (BGR)
        gradient_thresh: Seuil pour la détection de gradient
        min_size: Taille minimale d'un bloc vide
        smooth: Marge pour adoucir les limites
        recursive: Si True, applique récursivement l'algorithme sur chaque bloc
        depth: Profondeur actuelle de récursion
        max_depth: Profondeur maximale de récursion
    
    Returns:
        Liste de tuples (x, y, w, h) représentant les coordonnées des blocs
    """  # noqa: D205
    # Obtenir les dimensions de l'image
    h, w = image.shape[:2]
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculer les gradients
    gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)
    gradient = np.maximum(gradient_x, gradient_y)
    gradient[gradient < 50] = 0  # Supprimer le bruit de fond
    
    # Calculer les projections de gradient
    h_projection = np.sum(gradient, axis=0)  # Projection horizontale
    v_projection = np.sum(gradient, axis=1)  # Projection verticale
    
    # Trouver les séparations horizontales et verticales
    h_separations = find_separations(h_projection, gradient_thresh, min_size, smooth)
    v_separations = find_separations(v_projection, gradient_thresh, min_size, smooth)
    
    # Si aucune séparation n'est trouvée, retourner l'image entière
    if h_separations is None or v_separations is None:
        return [(0, 0, w, h)]
    
    # Convertir les séparations en régions
    h_regions = separations_to_regions(h_separations, w)
    v_regions = separations_to_regions(v_separations, h)
    
    # Créer la liste des coordonnées des blocs
    blocks = []
    for y_start, y_end in v_regions:
        for x_start, x_end in h_regions:
            # Vérifier que le bloc a une taille significative
            if (x_end - x_start) > 5 and (y_end - y_start) > 5:
                # Si récursif et pas atteint la profondeur maximale
                if recursive and depth < max_depth:
                    # Extraire le bloc pour l'analyse récursive
                    sub_image = image[y_start:y_end, x_start:x_end]
                    
                    # Adapter le seuil en fonction de la taille du bloc
                    sub_thresh = gradient_thresh * (sub_image.shape[0] * sub_image.shape[1]) / (h * w)
                    
                    # Détecter les sous-blocs
                    sub_blocks = detect_image_blocks(
                        sub_image, gradient_thresh=sub_thresh, 
                        min_size=min_size, smooth=smooth,
                        recursive=recursive, depth=depth+1, max_depth=max_depth
                    )
                    
                    # Si aucun sous-bloc n'est trouvé ou un seul qui couvre tout
                    if len(sub_blocks) == 1 and sub_blocks[0] == (0, 0, sub_image.shape[1], sub_image.shape[0]):
                        blocks.append((x_start, y_start, x_end - x_start, y_end - y_start))
                    else:
                        # Ajuster les coordonnées des sous-blocs par rapport à l'image d'origine
                        for sx, sy, sw, sh in sub_blocks:
                            blocks.append((x_start + sx, y_start + sy, sw, sh))
                else:
                    blocks.append((x_start, y_start, x_end - x_start, y_end - y_start))
    
    return blocks

def find_separations(projection, threshold, min_size=10, smooth=1):
    """Trouve les séparations dans une projection de gradient.
    
    Args:
        projection: Projection du gradient (somme par ligne ou colonne)
        threshold: Seuil pour considérer une valeur comme significative
        min_size: Taille minimale d'une séparation
        smooth: Marge pour adoucir les limites
    
    Returns:
        Liste de séparations [début, fin]
    """
    # Trouver les indices où la projection dépasse le seuil
    indices = np.where(projection > threshold)[0]
    
    if len(indices) == 0:
        return None
    
    # Initialiser la liste des séparations
    separations = []
    
    # Ajouter le début si nécessaire
    if indices[0] > min_size:
        start = 0
        end = max(0, indices[0] - smooth)
        separations.append([start, end])
    
    # Trouver les blocs vides entre les zones d'intérêt
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > min_size:
            start = min(len(projection), indices[i] + smooth)
            end = max(0, indices[i + 1] - smooth)
            if start < end:
                separations.append([start, end])
    
    # Ajouter la fin si nécessaire
    if len(projection) - indices[-1] > min_size:
        start = min(len(projection), indices[-1] + smooth)
        end = len(projection)
        separations.append([start, end])
    
    return separations

def separations_to_regions(separations, size):
    """Convertit les séparations en régions.
    
    Args:
        separations: Liste de séparations [début, fin]
        size: Taille totale (largeur ou hauteur)
    
    Returns:
        Liste de régions [début, fin]
    """
    regions = []
    
    # Cas particulier: aucune séparation
    if len(separations) == 0:
        regions.append([0, size])
        return regions
    
    # Premier bloc si nécessaire
    if separations[0][0] > 0:
        regions.append([0, separations[0][0]])
    
    # Blocs intermédiaires
    for i in range(len(separations) - 1):
        regions.append([separations[i][1], separations[i + 1][0]])
    
    # Dernier bloc si nécessaire
    if separations[-1][1] < size:
        regions.append([separations[-1][1], size])
    
    return regions

def upscale_image(image, scale_factor=2.0, method='lanczos4'):
    """Redimensionne une image avec une méthode optimisée pour le texte et les détails.
    
    Args:
        image: Image à redimensionner
        scale_factor: Facteur d'agrandissement
        method: Méthode d'interpolation ('nearest', 'linear', 'cubic', 'lanczos4', 'waifu2x')
    
    Returns:
        Image redimensionnée
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if method == 'nearest':
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    elif method == 'linear':
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif method == 'cubic':
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif method == 'lanczos4':
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Méthode par défaut pour le texte (Lanczos)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Amélioration supplémentaire pour le texte (optionnelle)
    if method == 'text_enhance':
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Reconversion en couleur
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Blend avec l'original pour conserver de la couleur
        alpha = 0.7
        resized = cv2.addWeighted(resized, alpha, enhanced_color, 1-alpha, 0)
    
    return resized

def sharpen_image(image):
    """Applique un filtre de netteté à l'image pour améliorer la lisibilité du texte.
    
    Args:
        image: Image à améliorer
    
    Returns:
        Image améliorée
    """
    # Création du kernel de netteté
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Application du filtre
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def extract_and_upscale_blocks(image_path, output_dir="output_blocks", max_depth=2, scale_factor=2.0, 
                              upscale_method='lanczos4', sharpen=True, save_quality=95):
    """Détecte, extrait, redimensionne et sauvegarde les blocs d'une image avec une haute qualité.
    
    Args:
        image_path: Chemin vers l'image d'entrée
        output_dir: Dossier de sortie pour les sous-images
        max_depth: Profondeur maximale de récursion
        scale_factor: Facteur d'agrandissement
        upscale_method: Méthode de redimensionnement
        sharpen: Si True, applique un filtre de netteté
        save_quality: Qualité de sauvegarde (0-100) pour les images JPEG
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return
    
    # Adapter le seuil en fonction de la taille de l'image
    thresh = adaptive_threshold(image)
    
    # Détecter les blocs avec récursion
    blocks = detect_image_blocks(
        image, gradient_thresh=thresh, min_size=10, smooth=1,
        recursive=True, depth=0, max_depth=max_depth
    )
    
    # Sauvegarder les blocs avec haute qualité et redimensionnement
    blocks_paths = []
    for i, (x, y, w, h) in enumerate(blocks):
        # Extraire le bloc directement de l'image originale
        block_img = image[y:y+h, x:x+w]
        
        # Redimensionner le bloc
        upscaled_img = upscale_image(block_img, scale_factor, upscale_method)
        
        # Appliquer un filtre de netteté si demandé
        if sharpen:
            upscaled_img = sharpen_image(upscaled_img)
        
        # Déterminer le format de sortie
        _, ext = os.path.splitext(image_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            # Pour JPEG, utiliser la qualité spécifiée
            output_path = os.path.join(output_dir, f"bloc_{i+1}_upscaled.jpg")
            cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_JPEG_QUALITY, save_quality])
        elif ext.lower() == '.png':
            # Pour PNG, utiliser la compression maximale
            output_path = os.path.join(output_dir, f"bloc_{i+1}_upscaled.png")
            cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # Pour les autres formats, utiliser PNG sans perte
            output_path = os.path.join(output_dir, f"bloc_{i+1}_upscaled.png")
            cv2.imwrite(output_path, upscaled_img)

        blocks_paths.append(output_path)
    
    print(f"{len(blocks)} blocs ont été détectés, agrandis par un facteur {scale_factor} "
          f"avec la méthode '{upscale_method}' et sauvegardés dans le dossier {output_dir}")

    return blocks, blocks_paths

def adaptive_threshold(image, initial_thresh=6000, min_size=10):
    """Détermine automatiquement le seuil optimal pour la détection des blocs.
    
    Args:
        image: Image d'entrée
        initial_thresh: Seuil initial
        min_size: Taille minimale d'un bloc
    
    Returns:
        Seuil adapté
    """
    h, w = image.shape[:2]
    area = h * w
    
    # Adapter le seuil en fonction de la taille de l'image
    if area > 1000000:  # Grande image (> 1MP)
        return initial_thresh * 2
    elif area < 100000:  # Petite image (< 0.1MP)
        return initial_thresh / 2
    else:
        return initial_thresh

def process_image_with_block_ocr(image_path: str, output_dir: str = "output_blocks", 
                               tesseract_config: str = r'--oem 3 --psm 6 -l fra') -> Dict:
    """Découpe une image en blocs, applique l'OCR sur chaque bloc,
    et combine les résultats en les ramenant aux coordonnées de l'image originale.
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return {"words": [], "bboxes": [], "confidences": []}
    
    # Dimensions de l'image originale
    image_height, image_width = image.shape[:2]
    
    # Détecter les blocs avec récursion
    blocks, blocks_paths = extract_and_upscale_blocks(image_path, 
        scale_factor=3.0,  # Facteur d'agrandissement (3x)
        upscale_method='lanczos4',  # Méthode optimisée pour le texte
        sharpen=True,  # Appliquer un filtre de netteté
        save_quality=100  # Qualité maximale pour JPEG
    )
    
    # Initialiser le dictionnaire de résultats
    extracted_data: Dict[str, List[Any]] = {
        "words": [],       # List[str] à la rigueur si vous voulez être plus précis
        "bboxes": [],      # List[Tuple[int, int, int, int]]
        "confidences": []  # List[int] ou List[float]
    }
    
    # Facteur de mise à l'échelle pour compenser l'agrandissement
    scale_factor = 3.0
    
    for block, block_path in zip(blocks, blocks_paths, strict=False):
        # Extraire les coordonnées du bloc
        x, y, w, h = block
        
        # Appliquer l'OCR avec Tesseract
        block_data = pytesseract.image_to_data(
            block_path, output_type=pytesseract.Output.DICT, config=tesseract_config
        )

        # Parcourir les mots détectés
        for j in range(len(block_data['text'])):
            # Ignorer les entrées vides
            if not block_data['text'][j].strip():
                continue
            
            # Récupérer les coordonnées locales au bloc (ajustées pour l'échelle)
            local_x = int(block_data['left'][j] / scale_factor)
            local_y = int(block_data['top'][j] / scale_factor)
            local_w = int(block_data['width'][j] / scale_factor)
            local_h = int(block_data['height'][j] / scale_factor)
            
            # Convertir en coordonnées globales
            global_x = x + local_x
            global_y = y + local_y
            
            # Créer la bbox au format x1,y1,x2,y2 (SANS inverser les coordonnées Y)
            bbox_tuple = (
                global_x, 
                image_height - (global_y + local_h), 
                global_x + local_w, 
                image_height - global_y
            )
            
            # Ajouter aux données
            extracted_data["words"].append(block_data['text'][j])
            extracted_data["bboxes"].append(bbox_tuple)
            extracted_data["confidences"].append(block_data['conf'][j])
    
    return extracted_data

def process_ocr(image_path):
    """Traite l'OCR en découpant d'abord l'image en blocs puis en appliquant l'OCR sur chaque bloc."""
    try:
        tesseract_config = r'--oem 3 --psm 6 -l fra' # config pour le français et les documents administratifs
        extracted_data = process_image_with_block_ocr(
            image_path, 
            tesseract_config=tesseract_config,
        )
        converted_data = convert_nested_numpy_types(extracted_data)
        return converted_data
    
    except Exception as e:
        print(f"Erreur lors de l'extraction avec Tesseract: {e}")
        return {}

def normalize_bboxes(bboxes, image_size, scale=1000):
    """Normaliser les bounding boxes à l'échelle 0-1000 (format attendu par LayoutLMv3).
    
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

def predict_with_model(image_path):
    """
    Effectue la prédiction sur une image en utilisant la méthode du script d'entraînement
    """
    try:
        # Extraire le texte et les bounding boxes avec plusieurs OCR
        ocr_data = process_ocr(image_path)
        
        if not ocr_data["words"]:
            print("No OCR data detected")
            return [], [], []
        
        # Lire l'image originale
        img = PILImage.open(image_path).convert("RGB")  # Forcer le format RGB
        img_np = np.array(img)  # Convertir en numpy pour s'assurer que le format est correct
        img = PILImage.fromarray(img_np)  # Reconvertir en PIL Image
        width, height = img.size
        
        # Normaliser les bounding boxes
        norm_boxes = normalize_bboxes(ocr_data["bboxes"], (width, height))
        
        # Préparer les données pour le modèle - même format que le script d'entraînement
        encoding = processor(
            images=img,
            text=[ocr_data["words"]],
            boxes=[norm_boxes],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Déplacer les données vers le device
        for k, v in encoding.items():
            encoding[k] = v.to(device)
        
        # Effectuer la prédiction en mode évaluation
        with torch.no_grad():
            outputs = model(**encoding)
            
        # Traiter les prédictions - même méthode que compute_metrics dans le script d'entraînement
        predictions = outputs.logits.argmax(dim=2)
        
        # Récupérer les labels prédits
        predicted_labels = []
        word_ids = encoding.word_ids(batch_index=0)
        
        # Extraire les prédictions pour chaque mot (premier token de chaque mot)
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                label_id = predictions[0, idx].item()
                predicted_labels.append(id2label[label_id])
                prev_word_id = word_id
        
        # S'assurer que nous avons le même nombre de prédictions que de mots
        words = ocr_data["words"]
        boxes = ocr_data["bboxes"]
        
        # En cas de désynchronisation entre le nombre de mots et de prédictions
        min_len = min(len(words), len(predicted_labels))
        words = words[:min_len]
        boxes = boxes[:min_len]
        predicted_labels = predicted_labels[:min_len]
        
        print(f"Predicted {len(predicted_labels)} labels for {len(words)} words")
        
        return words, boxes, predicted_labels
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

def merge_bio_boxes(words, boxes, labels):
    """
    Fusionne les boîtes avec les étiquettes B-/I- consécutives pour le même type d'entité
    """
    if not words or not boxes or not labels:
        return [], [], []
        
    merged_words = []
    merged_boxes = []
    merged_labels = []
    
    i = 0
    while i < len(labels):
        current_label = labels[i]
        
        # Si le label actuel est "O" ou ne commence pas par "B-", l'ajouter simplement
        if current_label == "O":
            merged_words.append(words[i])
            merged_boxes.append(boxes[i])
            merged_labels.append(current_label)
            i += 1
            continue
            
        # Si c'est un début d'entité (B-)
        if current_label.startswith("B-"):
            entity_type = current_label[2:]  # Extraire le type d'entité
            entity_words = [words[i]]
            
            # Initialiser la boîte fusionnée avec la boîte courante
            x1, y1, x2, y2 = boxes[i]
            
            # Chercher les tokens I- consécutifs pour cette entité
            j = i + 1
            while j < len(labels) and labels[j].startswith("I-") and labels[j][2:] == entity_type:
                entity_words.append(words[j])
                
                # Mettre à jour la boîte fusionnée pour englober la boîte courante
                bx1, by1, bx2, by2 = boxes[j]
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)
                
                j += 1
            
            # Ajouter l'entité fusionnée
            merged_words.append(" ".join(entity_words))
            merged_boxes.append((x1, y1, x2, y2))
            merged_labels.append(entity_type)  # Stocker le type d'entité sans le préfixe B-/I-
            
            # Passer à la prochaine entité
            i = j
        else:
            # Si c'est un I- orphelin, le traiter comme un token séparé
            if current_label.startswith("I-"):
                entity_type = current_label[2:]
                merged_words.append(words[i])
                merged_boxes.append(boxes[i])
                merged_labels.append(entity_type)  # Stocker le type d'entité sans le préfixe I-
            # Sinon (cas peu probable), le conserver tel quel
            else:
                merged_words.append(words[i])
                merged_boxes.append(boxes[i])
                merged_labels.append(current_label)
            
            i += 1
    
    return merged_words, merged_boxes, merged_labels

def post_process_merged_entities(merged_words, merged_boxes, merged_labels):
    """
    Post-traitement des entités fusionnées pour corriger les chevauchements
    et maximiser la précision des prédictions
    """
    if not merged_words:
        return merged_words, merged_boxes, merged_labels
    
    # Créer des dictionnaires pour stocker les entités par type
    entities_by_type = {}
    for i, (word, box, label) in enumerate(zip(merged_words, merged_boxes, merged_labels)):
        if label == "O":
            continue
            
        if label not in entities_by_type:
            entities_by_type[label] = []
            
        entities_by_type[label].append((word, box, i))
    
    # Vérifier et fusionner les entités proches du même type
    for label, entities in entities_by_type.items():
        if len(entities) <= 1:
            continue
            
        # Trier les entités par position horizontale
        entities.sort(key=lambda x: x[1][0])  # Trier par x1
        
        i = 0
        while i < len(entities) - 1:
            # Vérifier si deux entités consécutives sont proches
            word1, box1, idx1 = entities[i]
            word2, box2, idx2 = entities[i+1]
            
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculer la distance horizontale
            h_distance = x1_2 - x2_1
            
            # Si les entités sont proches ou se chevauchent
            if h_distance < 50:  # Seuil de distance en pixels
                # Fusionner les deux entités
                new_word = f"{word1} {word2}"
                new_box = (
                    min(x1_1, x1_2),
                    min(y1_1, y1_2),
                    max(x2_1, x2_2),
                    max(y2_1, y2_2)
                )
                
                # Mettre à jour l'entité actuelle
                entities[i] = (new_word, new_box, idx1)
                
                # Marquer l'entité suivante pour suppression
                merged_words[idx2] = None
                merged_boxes[idx2] = None
                merged_labels[idx2] = None
                
                # Supprimer l'entité de notre liste
                entities.pop(i+1)
            else:
                i += 1
    
    # Filtrer les entités marquées pour suppression
    filtered_words = []
    filtered_boxes = []
    filtered_labels = []
    
    for word, box, label in zip(merged_words, merged_boxes, merged_labels):
        if word is not None:
            filtered_words.append(word)
            filtered_boxes.append(box)
            filtered_labels.append(label)
    
    return filtered_words, filtered_boxes, filtered_labels

def process_image(image_path):
    """
    Traite l'image et crée une visualisation avec les entités détectées
    """
    # Prédire les labels
    words, boxes, labels = predict_with_model(image_path)
    
    if not words:
        return None, "Aucun texte détecté dans l'image."
    
    # Fusionner les boîtes BIO
    merged_words, merged_boxes, merged_labels = merge_bio_boxes(words, boxes, labels)
    
    # Post-traitement pour améliorer les résultats
    merged_words, merged_boxes, merged_labels = post_process_merged_entities(merged_words, merged_boxes, merged_labels)
    
    # Lire l'image originale
    img = PILImage.open(image_path)
    img_arr = np.array(img)
    img_height, img_width = img_arr.shape[:2]
    
    # Créer une figure pour la visualisation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr)
    ax.axis('off')
    
    # Dictionnaire pour stocker les entités trouvées
    found_entities = {}
    
    # Ajouter les boîtes fusionnées
    for i, (word, box, label) in enumerate(zip(merged_words, merged_boxes, merged_labels)):
        # Ne pas afficher les labels "O"
        if label == "O":
            continue
        
        # Trouver la couleur appropriée pour ce type de label
        color = "red"  # Couleur par défaut
        for key, value in COLOR_MAP.items():
            if label.startswith(key):
                color = value
                break
        
        x1, y1, x2, y2 = box
        
        # Ajouter l'entité au dictionnaire
        if label not in found_entities:
            found_entities[label] = []
        found_entities[label].append(word)
        
        # Créer un rectangle
        rect = Rectangle(
            (x1, img_height - y2),  # Inverser y2 car l'axe y de matplotlib part du bas
            (x2 - x1), 
            (y2 - y1), 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none', 
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Ajouter un label
        ax.text(
            x1, 
            img_height - y1,  # Inverser y1 car l'axe y de matplotlib part du bas
            f"{label}", 
            color='white', 
            fontsize=10, 
            bbox=dict(facecolor=color, alpha=0.7)
        )
    
    # Sauvegarder la figure en tant qu'image
    fig.tight_layout()
    result_path = os.path.join(RESULTS_FOLDER, os.path.basename(image_path))
    plt.savefig(result_path, bbox_inches='tight')
    plt.close(fig)
    
    # Convertir en base64 pour l'affichage HTML
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 10))
    plt.imshow(plt.imread(result_path))
    plt.axis('off')
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    # Encoder en base64
    b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return b64_image, found_entities

# Routes Flask
@app.route('/')
def index():
    return render_template('index_inference.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier téléchargé'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'})
    
    if file:
        # Sauvegarder le fichier
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Traiter l'image
        try:
            b64_image, entities = process_image(file_path)
            
            if not b64_image:
                return jsonify({'error': 'Aucun texte détecté dans l\'image'})
            
            # Préparer les résultats
            results = []
            for label, words in entities.items():
                results.append({
                    'label': label,
                    'words': words,
                    'color': COLOR_MAP.get(label.split('.')[0] if '.' in label else label, 'red')
                })
            
            return jsonify({
                'success': True,
                'image': b64_image,
                'entities': results
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Erreur lors du traitement de l\'image: {str(e)}'})
    
    return jsonify({'error': 'Une erreur est survenue lors du téléchargement du fichier'})


if __name__ == '__main__':
    print("Starting Text Annotation Tool on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)