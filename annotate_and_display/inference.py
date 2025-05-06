import os
import json
import numpy as np
import torch
import cv2
import pytesseract
from PIL import Image as PILImage
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from docling.document_converter import DocumentConverter
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non interactif
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import easyocr
import Levenshtein

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = '../layoutlmv3_ft/results/final_model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

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
    "DATE": "green",
    "TYPE": "green",
    "MEDECINE": "green"
}

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
    
    # Amélioration du contraste par égalisation d'histogramme adaptative
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    # Binarisation adaptative pour améliorer la détection du texte
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Créer un nom pour l'image améliorée
    enhanced_path = os.path.join(UPLOAD_FOLDER, f"enhanced_{os.path.basename(image_path)}")
    
    # Sauvegarder l'image améliorée
    cv2.imwrite(enhanced_path, binary)
    
    return enhanced_path

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

def is_new_bbox(new_bbox, existing_bboxes, overlap_threshold=0.5):
    """
    Vérifie si une bounding box est nouvelle (pas suffisamment de chevauchement avec les existantes)
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

def combine_ocr_results(image_path):
    """
    Combiner les résultats OCR de EasyOCR, Tesseract et Docling
    """
    try:
        print(f"Extracting text from {image_path} with multiple OCR engines...")
        
        # 1. Utiliser EasyOCR en premier (souvent plus précis)
        easyocr_data = extract_text_with_easyocr(image_path)
        
        # 2. Utiliser Tesseract
        tesseract_data = extract_text_with_tesseract(image_path)
        
        # 3. Utiliser Docling 
        docling_data = extract_text_with_docling(image_path)
        
        # 4. Combiner les résultats en gardant EasyOCR comme principale source
        combined_data = {
            "words": easyocr_data["words"].copy(),
            "bboxes": easyocr_data["bboxes"].copy(),
            "page_numbers": [1] * len(easyocr_data["words"]),
            "engine": ["easyocr"] * len(easyocr_data["words"])
        }
        
        # 5. Ajouter les résultats de Tesseract qui ne chevauchent pas ceux de EasyOCR
        for i, (word, bbox) in enumerate(zip(tesseract_data["words"], tesseract_data["bboxes"])):
            if is_new_bbox(bbox, combined_data["bboxes"], overlap_threshold=0.3):
                # Vérifier que le mot n'est pas vide ou un simple caractère
                if len(word.strip()) > 1:
                    combined_data["words"].append(word)
                    combined_data["bboxes"].append(bbox)
                    combined_data["page_numbers"].append(1)
                    combined_data["engine"].append("tesseract")
        
        # 6. Ajouter les résultats de Docling qui ne chevauchent pas ceux déjà présents
        for i, (word, bbox) in enumerate(zip(docling_data["words"], docling_data["bboxes"])):
            if is_new_bbox(bbox, combined_data["bboxes"], overlap_threshold=0.3):
                if len(word.strip()) > 1:
                    combined_data["words"].append(word)
                    combined_data["bboxes"].append(bbox)
                    combined_data["page_numbers"].append(docling_data["page_numbers"][i] if "page_numbers" in docling_data else 1)
                    combined_data["engine"].append("docling")
        
        print(f"Found {len(combined_data['words'])} words total: "
              f"{len(easyocr_data['words'])} from EasyOCR, "
              f"{len([e for e in combined_data['engine'] if e == 'tesseract'])} additional from Tesseract, "
              f"{len([e for e in combined_data['engine'] if e == 'docling'])} additional from Docling")
        
        return combined_data
        
    except Exception as e:
        print(f"Error processing OCR: {e}")
        import traceback
        traceback.print_exc()
        return {"words": [], "bboxes": [], "page_numbers": [], "engine": []}

def normalize_bboxes(bboxes, image_size, scale=1000):
    """
    Normalise les bounding boxes à l'échelle 0-1000 (format attendu par LayoutLMv3)
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
        ocr_data = combine_ocr_results(image_path)
        
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