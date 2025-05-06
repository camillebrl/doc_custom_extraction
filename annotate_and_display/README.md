## ui.py – Annotation interactive et génération de données

```mermaid
flowchart LR
    A[Chargement du document] --> B[OCR combiné<br/>(EasyOCR, Tesseract, Docling)]
    B --> C[Interface web Flask<br/>pour sélectionner des bboxes]
    C --> D[Sélection manuelle des bbox<br/>et choix de labels]
    D --> E[Enregistrement des annotations<br/>temp_annot.jsonl]
    E --> F[Génération automatique d’augmentations<br/>(contraste, bruit, rotation…)]
    F --> G[Re-OCR sur images augmentées]
    G --> H[Ré-ajustement des bbox et sauvegarde JSONL]
```

Dans **ui.py**, on propose une application Flask permettant d’annoter **manuellement** des données visuelles en **sélectionnant** directement les bounding boxes détectées par OCR et en choisissant **librement** les labels (totalement customisables) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

- **But principal** : créer un fichier `temp_annot.jsonl` qui servira au **finetuning** du modèle KIE (LayoutLMv3 ou équivalent).  
- **OCR multicouche** : trois moteurs (EasyOCR, Tesseract, Docling) sont appliqués séquentiellement puis combinés sans chevauchement pour obtenir la reconstruction la plus **précise** possible des mots et de leurs boîtes :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}.   
- **Annotation manuelle** :  
  - L’utilisateur sélectionne dans l’interface **toutes** les bounding boxes qu’il souhaite annoter, puis entre le label de son choix (100 % customisable).  
  - Le système génère automatiquement des **tags BIO** :  
    - `B-<LABEL>` pour le premier bbox d’une entité  
    - `I-<LABEL>` pour chaque bbox suivant  
  - **Exemple** : l’OCR a fragmenté le nom `Monsieur Patate LTD` en trois bboxes.  
    1. Sélectionner les trois  
    2. Choisir le label `NAME`  
    3. Stockage dans `temp_annot.jsonl` :  
    ```json
    [
      { "bbox": [[x1, y1, x2, y2], [x3, y3, x4, y4], [x5, y5, x6, y6]], "words": ["Monsieur" , "Patate" "LTD", "lives" "in", "Toulouse"], "label": ["B-NAME", "I-NAME", "I-NAME", "O", "O", "B-City"]}
    ]
    ```  
  - Si une même entité est éclatée en plusieurs morceaux par l’OCR, il suffit de sélectionner **tous** les morceaux pour qu’ils soient étiquetés ensemble (l'un après l'autre, avant de valider la labélisation); l’utilisateur clique-glisse pour sélectionner **toutes** les boxes formant une entité (même si l’OCR l’a découpée en plusieurs morceaux) et lui associe un label. Le système génère automatiquement des tags **BIO** („B-” pour le début, „I-” pour la suite).  
   - Exemple : pour un nom en 3 boxes (« Monsieur » « Patate » « LTD »), on sélectionne les trois, on choisit le label `NAME` et on obtient dans `temp_annot.jsonl` :  
     ```json
     ["B-NAME", "I-NAME", "I-NAME"]
     ```  
  - Note que le label "O" sert de "N/A" (non-applicable). Ce sont les mots du documents qu'on n'a pas labélisés. Eh oui, parce que la tâche de KIE sur des documents est en fait une tâche de token classification, cad qu'il faut classifier CHAQUE token, même si tous les tokens ne nous intéressent pas!   
- **Augmentation et réajustement** : une fois les annotations sauvegardées dans `temp_annot.jsonl` (via le bouton **Save to Annotation File**), ui.py applique des transformations d’image (contraste, bruit, rotation légère, perspective, taches d’encre, etc.), relance l’OCR sur chaque image modifiée, et **réajuste** les annotations par transfert de labels basé sur la similarité de texte :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.  
  1. Application de transformations d’image (contraste, bruit, rotation, perspective, taches…)  
  2. Nouvelle passe OCR sur chaque image modifiée  
  3. Transfert automatique des labels : rapprochement des textes pour **réajuster** les bboxes  
- **Sortie finale** : le fichier `temp_annot.jsonl`, normalisé pour LayoutLMv3 (bboxes à l’échelle 0–1000), servira ensuite de jeu de données pour le **finetuning** du modèle KIE.

---

## inference.py – Chargement du modèle fine-tuned et prédictions

Le script **inference.py** ne fonctionne **qu’après** le fine-tuning lancé dans le dossier `layoutlmv3_ft` (après clic sur **Clear and finish** dans ui.py, on exécute le notebook/train script qui produit `results/final_model`) :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}.  
1. **Chargement du modèle** : on récupère `label_mappings.json` et le modèle LayoutLMv3ForTokenClassification entraîné, via `LayoutLMv3Processor` (avec OCR désactivé), sur le device CPU/GPU disponible :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}.  
2. **Pipeline de prédiction** :  
   - Téléversement d’une image via l’interface Flask.  
   - **OCR combiné** : Docling d’abord, puis Tesseract pour ajouter les mots manquants.  
   - **Normalisation** des boîtes au format 0–1000 attendu par LayoutLMv3.  
   - **Inférence** : passage dans le modèle pour obtenir un label par token, puis agrégation des premiers tokens de chaque mot.  
3. **Post-traitement** :  
   - **Fusion BIO** : on regroupe séquentiellement les a
