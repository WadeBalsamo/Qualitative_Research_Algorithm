"""
embedding.py
------------

Loads the fine-tuned Mental-BERT model and tokenizer. Provides functions to encode sentences into dense vector embeddings using the QLoRA-adapted model. Supports batch encoding and GPU acceleration. Ensures compatibility with sentence-level input and outputs for graph construction.
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from typing import List, Optional, Any
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

class MentalBERTEmbedder:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, lora_weights_path: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if lora_weights_path is not None:
            # Load LoRA/QLoRA weights if provided
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode_sentences(self, sentences: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        Encodes a list of sentences into embeddings using Mental-BERT.
        Returns a tensor of shape (num_sentences, embedding_dim).
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                # L2 normalization for consistent similarity
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str], labels: Optional[List[int]] = None, tokenizer: Any = None, max_length: int = 128) -> None:
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self) -> int:
        return len(self.sentences)
    def __getitem__(self, idx: int) -> dict:
        item = self.tokenizer(self.sentences[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in item.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class MentalBERTTrainer:
    def __init__(self, model_name: str, num_labels: int, device: Optional[str] = None, lora_config: Optional[LoraConfig] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def train(self, train_texts: List[str], train_labels: List[int], val_texts: List[str], val_labels: List[int], epochs: int = 3, batch_size: int = 16, lr: float = 2e-5) -> None:
        train_dataset = SentenceDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentenceDataset(val_texts, val_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_mentalbert.pt')
    def load_best(self, path='models/best_mentalbert.pt'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
    def predict(self, texts, batch_size=16):
        dataset = SentenceDataset(texts, tokenizer=self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        return preds
