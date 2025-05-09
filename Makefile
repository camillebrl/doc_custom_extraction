SHELL := /bin/bash

.PHONY: help deps annotate_data finetuning_model final_inference all

help:
	@echo "Usage:"
	@echo "  make deps        		# installer les paquets système"
	@echo "  make annotate_data     # lancer le script d'annotation des données (app flask)+ augmentation des données. Une fois l'annotation terminée, cliquer sur le bouton 'clear and finish' de l'app."
	@echo "  make finetuning_model  # lancer le fine-tuning de layoutlmv3 sur les données annotées + augmentées"
	@echo "  make final_inference   # lancer l'ui de l'inference du modèle fine-tuner pour le tester"
	@echo "  make all         		# tout faire dans l’ordre"

deps:
	sudo apt-get update
	sudo apt-get install -y tesseract-ocr libtesseract-dev \
	                       tesseract-ocr-eng tesseract-ocr-fra

annotate_data:
	cd annotate_and_display
	poetry install
	poetry run python ui.py

finetuning_model:
	cd ../layoutlmv3_ft
	poetry install
	poetry run python layoutlmv3_ft.py

final_inference:
	cd ../annotate_and_display
	poetry install
	poetry run python annotate.py

all:
	deps annotate_data finetuning_model final_inference