"""
classification.py
-----------------

Implements transparent, graph-based classification of sentences and communities. Combines Mental-BERT construct predictions, graph structure, and community-level aggregation to assign therapeutic construct labels. Provides confidence scores, ranked similar sentences, and supports uncertainty quantification for interpretability.
"""

import numpy as np
from typing import List, Dict, Any

def classify_sentences(embeddings: np.ndarray, model: Any, label_names: List[str]) -> List[Dict[str, Any]]:
    """
    Uses the fine-tuned Mental-BERT model to predict construct labels for each sentence embedding.
    Returns a list of dicts with label, confidence, and logits for each sentence.
    Ensures transparency by returning all probabilities for interpretability.
    """
    import torch
    with torch.no_grad():
        logits = model.classifier(torch.tensor(embeddings).to(model.device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    results = []
    for i, p in enumerate(probs):
        label_idx = int(np.argmax(p))
        results.append({
            'label': label_names[label_idx],
            'confidence': float(p[label_idx]),
            'probs': p.tolist()
        })
    return results

def aggregate_community_labels(partition: Dict[int, int], sentence_labels: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Aggregates sentence-level labels to assign a dominant label to each community.
    Returns a dict mapping community id to label.
    """
    from collections import Counter
    comm_labels = {}
    for node, comm_id in partition.items():
        if comm_id not in comm_labels:
            comm_labels[comm_id] = []
        comm_labels[comm_id].append(sentence_labels[node]['label'])
    return {cid: Counter(labels).most_common(1)[0][0] for cid, labels in comm_labels.items()}
