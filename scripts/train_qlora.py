"""
train_qlora.py
--------------

Script for parameter-efficient fine-tuning of the Mental-BERT model using QLoRA on expert-coded therapeutic transcript data, for use in the AQUA+GNN pipeline.
"""

import os
import logging
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from mentalbert_sentence_aqua import preprocessing, config
from mentalbert_sentence_aqua.embedding import SentenceDataset

logging.basicConfig(level=logging.INFO)

# Example config
MODEL_NAME = 'mental/mental-bert-base-uncased'
NUM_LABELS = 5
QLORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'task_type': TaskType.SEQ_CLS
}

def load_srt_data(data_dir: str, label_map: Dict[str, int]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.srt'):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            sents = preprocessing.parse_srt_file(fpath)
            for s in sents:
                if s.label and s.label.lower() in label_map:
                    texts.append(preprocessing.normalize_text(s.text))
                    labels.append(label_map[s.label.lower()])
                else:
                    logging.warning(f"Skipping unlabeled or unknown label in {fname}: '{s.text[:30]}...' label='{s.label}'")
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")
    return texts, labels

# Load data from SRT files
data_dir: str = config.SRT_DATA_DIR
label_map: Dict[str, int] = {l.lower(): i for i, l in enumerate(config.CONSTRUCT_LABELS)}
train_texts, train_labels = load_srt_data(data_dir, label_map)
val_texts, val_labels = train_texts, train_labels  # TODO: Replace with real split

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
lora_config = LoraConfig(**QLORA_CONFIG)
model = get_peft_model(base_model, lora_config)

train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_val_loss = float('inf')
for epoch in range(15):
    model.train()
    for batch in train_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_mentalbert_qlora.pt')
print("QLoRA fine-tuning complete.")
