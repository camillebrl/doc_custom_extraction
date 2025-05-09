# Copyright 2025 Camille Barboule
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from seqeval.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments


def create_label_maps(categories):
    """
    Crée les mappings id2label et label2id à partir des catégories disponibles
    """
    labels = ["O"] + categories
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


def load_json_data(jsonl_file):
    """
    Charge manuellement un fichier JSONL en lisant ligne par ligne
    """
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "image_path" in item and not item["image_path"].startswith("../annotate_and_display/"):
                    item["image_path"] = os.path.join("../annotate_and_display", item["image_path"])
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Erreur de parsing JSON à la ligne: {e}")
                print(f"Ligne problématique: {line[:100]}...")
    return data


def prepare_dataset_from_jsonl(jsonl_file, split_ratio=0.8):
    data = load_json_data(jsonl_file)
    print(f"Nombre d'items chargés: {len(data)}")
    if data:
        print(f"Clés du premier item: {list(data[0].keys())}")
        print(f"Exemple de chemin d'image modifié: {data[0].get('image_path', 'Non disponible')}")

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    all_labels = []
    for item in data:
        all_labels.extend(item.get("labels", []))
    unique_labels = set(all_labels)
    unique_labels.discard("")
    sorted_labels = sorted(unique_labels)
    id2label, label2id = create_label_maps(sorted_labels)

    train_items, test_items = train_test_split(data, test_size=1-split_ratio, random_state=42)
    train_ds = CustomDataset(train_items)
    test_ds = CustomDataset(test_items)
    return train_ds, test_ds, id2label, label2id


class LayoutLMv3DataCollator:
    def __init__(self, processor, label2id, max_length=512):
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, features):
        images = [Image.open(f["image_path"]).convert("RGB") for f in features]
        texts = [f["words"] for f in features]
        boxes = [f["bboxes"] for f in features]
        labels = [f["labels"] for f in features]

        encoding = self.processor(
            images=images,
            text=texts,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        encoded_labels = []
        for i, word_labels in enumerate(labels):
            word_ids   = encoding.word_ids(batch_index=i)
            raw_ids    = [ self.label2id.get(word_labels[w], self.label2id["O"]) if w is not None else -100
                           for w in word_ids ]
            # indices des O
            O_pos      = [j for j, lab in enumerate(raw_ids) if lab == self.label2id["O"]]
            nonO_count = len(raw_ids) - len(O_pos)
            # ne conserver qu’un ratio 1:1
            keep_O     = set(random.sample(O_pos, min(len(O_pos), nonO_count)))
            filtered   = [ lab if (lab!=self.label2id["O"] or idx in keep_O) else -100
                           for idx, lab in enumerate(raw_ids) ]
            encoded_labels.append(torch.tensor(filtered))
        encoding["labels"] = torch.stack(encoded_labels)
        return encoding


def compute_metrics(p, id2label):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        seq_preds, seq_labels = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id == -100:
                continue
            seq_preds.append(id2label[p_id])
            seq_labels.append(id2label[l_id])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "accuracy": accuracy_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }


# class WeightedTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss = self.loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LayoutLMv3")
    parser.add_argument("--annotation_file", type=str, default="../annotate_and_display/temp_annot.jsonl")
    parser.add_argument("--output_dir", type=str, default="./results_v0")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    train_ds, test_ds, id2label, label2id = prepare_dataset_from_jsonl(
        args.annotation_file, split_ratio=args.split_ratio
    )
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    # Calcul des class_weights sur l'ensemble d'entraînement
    all_labels_flat = []
    for item in train_ds:
        all_labels_flat.extend([label2id.get(l, label2id["O"]) for l in item["labels"]])
    freq = Counter(all_labels_flat)
    total = sum(freq.values())
    num_labels = len(id2label)
    # Évite les divisions par zéro si une classe est absente
    class_weights = torch.tensor([total / freq.get(i, 1) for i in range(num_labels)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_labels

    # Initialisation du processor et modèle
    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # # Freeze du backbone
    # for param in model.layoutlmv3.parameters():
    #     param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # # Injection du loss_fct pondérée dans le Trainer
    # WeightedTrainer.loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=-100)

    collator = LayoutLMv3DataCollator(processor, label2id)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=args.logging_dir,
        report_to=["tensorboard"],
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval results:", metrics)

    # Sauvegarde du modèle final
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    with open(os.path.join(args.output_dir, "final_model", "label_mappings.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)


if __name__ == "__main__":
    main()
