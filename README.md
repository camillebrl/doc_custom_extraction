# Key Information Extraction (KIE)

## Aperçu de la tâche KIE
Key Information Extraction (KIE) consiste à détecter et à extraire automatiquement des éléments structurés (champs-clés, entités, paires clé-valeur) à partir de documents variés (formulaires, factures, reçus, etc.). Il s'agit en fait d'une tâche de classification multi-classes des mots issus de l'OCR. 

## Jeux de données et exemples de tâches
- **FUNSD** (Form Understanding in Noisy Scanned Documents) : extraction de paires clé-valeur depuis des formulaires annotés avec positions de tokens et catégories sémantiques.
- **SROIE** (Scanned Receipt OCR and Information Extraction) : identification et classification des champs clés (nom du magasin, total, TVA, date) sur des reçus de caisse.
- **CORD** (Complex Receipt Datasets) : version détaillée de reçus permettant l’extraction d’informations plus diversifiées et la reconnaissance de tables.
- **SOIE** (Scanned Outlines Information Extraction) : tâches d’extraction similaires sur relevés bancaires ou factures plus génériques.

Chaque dataset propose une **tâche de classification** (types de champs) et de **localisation** (boîtes englobantes), ou une **tâche générative** (générer directement le JSON de sortie).

## Familles de modèles pour la KIE
Deux grandes catégories de modèles s’affrontent sur ces tâches :

### 1. Modèles fine-tunés (add heads)
- **Principe** : partir d’un backbone pré-entraîné (LayoutLMv3, Donut, etc.), ajouter une tête spécialisée (classification, token classification) et fine-tuner sur la tâche cible.
- **Atouts** : légers, rapides à entraîner (quelques heures sur CPU ou petite GPU), nécessitent peu de ressources matérielles.
- **Exemples** :
  - **LayoutLMv3** : modèle multimodal traitant conjointement les tokens textuels, la mise en page (bboxes) et l’information visuelle extraite via une architecture Transformer unifiée.
  - **LILT (TILT)** : extension de LayoutLM pour la génération de sorties structurées à partir de tokens visuels et textuels, souvent utilisée en mode discriminatif.
- **Usage typique** : classification de tokens, extraction de paires clé-valeur via softmax sur chaque token.

### 2. Modèles génératifs (VLLMs)
- **Principe** : modèles de type « Vision + Language Large Models » qui reçoivent en entrée l’image du document et génèrent séquentiellement le JSON ou la liste des champs.
- **Atouts** : flexibles, peuvent gérer des sorties hétérogènes et imiter un assistant linguistique pour la documentation.
- **Exemple** :
  - **GenKIE** : génère directement les structures de sortie, robuste aux erreurs OCR.

> *Pour une revue détaillée des différentes familles de modèles KIE, voir le papier* [arXiv:2501.02235](https://arxiv.org/pdf/2501.02235).

## Workflow de l’outil
```mermaid
graph TD
  A[Annotate_and_display] --> B[ui.py]
  B --> C[layoutlmv3_ft.py]
  C --> D[Annotate_and_display (inférence)]
```

### 1. Annotate_and_display : préparation et annotation manuelle

- **Lancer l’interface** :
```shell
python annotate_and_display/ui.py
```

Exécutez cette commande depuis la racine du projet pour démarrer l’GUI.

Import des documents :

- Chargez vos PDFs ou images via l’interface.
- Le système segmente automatiquement les tokens et affiche les boîtes englobantes (bboxes).

Annotation interactive :

- Sélectionnez chaque token ou zone graphique.
- Attribuez une étiquette (ex : InvoiceDate, TotalAmount, VendorName).
- Sauvegardez vos annotations au format JSON compatible avec le fine-tuning.

Résultat : Un dossier annotations/ est généré, contenant un fichier JSON par document.

### 2. Fine-tuning du modèle multimodal (LayoutLMv3)
- **Lancer le script** :
```shell
python layoutlmv3_ft/layoutlmv3_ft.py
```
- Chargement du backbone :
    - Tokens textuels issus de l’OCR
    - Boîtes englobantes (bboxes normalisées)
    - Features visuelles extraites de l’image
- Tête spécialisée : classification de token pour extraire les paires clé-valeur
- Hyperparamètres : configurables via configs/layoutlmv3_config.yaml (learning rate, batch size, epochs)
- Matériel :
    - CPU (multi-threading) ou petite GPU (CUDA minimale)
    - Durée typique : ~2–4 h sur CPU, <1 h sur GPU modeste
- Sortie : Poids sauvegardés dans results/final_model/model.safetensors

### 3. Inférence et évaluation
- **Ré-exécuter l’interface** :
```shell
python annotate_and_display/inference.py
```
Chargement du modèle : L’interface charge automatiquement le modèle fine-tuné.
Le code ouvre une application, dans laquelle l'utilisateur peut charger un nouveau document. L’outil prédit les labels pour chaque token ou zone du document. Les prédictions sont affichées avec couleurs et scores de confiance.

## Installation de l'environnement
Tu auras avant tout besoin de Poetry (ou UV). Avant tout, assures-toi d'avoir une version de poetry > 2.0.0. Ensuite, ajoute la commande shell à poetry pour naviguer plus simplement dans l'environnement virtuel créé. 
```shell
poetry self add poetry-plugin-shell # à partir de poetry 2.x, poetry shell ne fonctionne plus, il faut ajouter ce plugin à poetry
```
Dans chacun des folders (annotate_and_display et layoutlmv3_ft), il te faudra faire une installation:
```shell
poetry install
poetry shell # on entre dans l'environnement virtuel temporaire créé par poetry
```
Une fois l'env activé, tu peux lancer les scripts! Tu auras l'endpoint des applications indiquées!