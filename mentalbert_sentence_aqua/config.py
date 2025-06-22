"""
config.py
---------

Configuration module for the Mental-BERT AQUA pipeline. Stores all hyperparameters, file paths, model settings, QLoRA fine-tuning parameters, and thresholds for graph construction and classification. Centralizes experiment settings for reproducibility and easy modification.
"""
# Example config structure
MODEL_NAME = "username/mentalbert"
FINETUNED_MODEL_PATH = "models/mentalbert-qlora"
SRT_DATA_DIR = "data/"
EMBEDDING_DIM = 768
QLORA_CONFIG = {
    'quantization': '4bit',
    'rank': 16,
    'alpha': 32,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'epochs': 15
}
SIMILARITY_THRESHOLD = 0.75
TEMPORAL_EDGE_DECAY = 0.9
CONSTRUCT_LABELS = [
    'attention_dysregulation',
    'experiential_avoidance',
    'attention_regulation',
    'metacognition',
    'reappraisal'
]
LORA_WEIGHTS_PATH = "models/best_mentalbert_qlora.pt"
CLUSTERING_METHOD = "louvain"  # Options: 'louvain', 'spectral', 'greedy'
