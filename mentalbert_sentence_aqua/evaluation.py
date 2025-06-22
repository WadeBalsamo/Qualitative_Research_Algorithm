"""
evaluation.py
-------------

Provides evaluation metrics for the pipeline:
- Inter-rater reliability (Cohen’s kappa) between model and human coders.
- Precision, recall, F1-score for each therapeutic construct.
- Graph modularity and community coherence metrics.
- Temporal consistency and progression analysis.
- Efficiency benchmarking (time savings vs. manual coding).
"""

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import networkx as nx
from typing import List, Dict

def compute_cohens_kappa(y_true: List[str], y_pred: List[str]) -> float:
    """
    Computes Cohen's kappa between true and predicted labels.
    """
    return cohen_kappa_score(y_true, y_pred)

def compute_precision_recall_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Computes precision, recall, and F1-score for each label.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    return {
        label: {'precision': float(p), 'recall': float(r), 'f1': float(f)}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }

def compute_modularity(G: nx.Graph, partition: Dict[int, int]) -> float:
    """
    Computes modularity of the graph partition.
    """
    import community as community_louvain
    return community_louvain.modularity(partition, G)
