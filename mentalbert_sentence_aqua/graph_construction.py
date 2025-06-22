"""
graph_construction.py
---------------------

Implements construction of the sentence graph for AQUA analysis. Nodes represent sentences; edges represent semantic similarity, temporal adjacency, and shared therapeutic constructs. Includes:
- Edge creation based on cosine similarity of embeddings (with thresholding).
- Temporal adjacency edges with decay weighting.
- Optional edges for shared construct predictions.
- Outputs a NetworkX graph object for clustering and visualization.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, LayerNorm

def build_sentence_graph(sentences: List[Any], embeddings: np.ndarray, similarity_threshold: float = 0.75, temporal_decay: float = 0.9, construct_labels: List[str] = None, confidences: List[float] = None, construct_conf_threshold: float = 0.7) -> nx.Graph:
    """
    Constructs a sentence graph where nodes are sentences and edges are based on semantic similarity and temporal adjacency.
    Args:
        sentences: List of SentenceUnit objects (from preprocessing).
        embeddings: np.ndarray of shape (num_sentences, embedding_dim).
        similarity_threshold: Cosine similarity threshold for semantic edges.
        temporal_decay: Decay factor for temporal adjacency edges.
    Returns:
        G: networkx.Graph object.
    """
    G = nx.Graph()
    num_sentences = len(sentences)
    # Add nodes
    for i, sent in enumerate(sentences):
        G.add_node(i, text=sent.text, speaker=sent.speaker, start=sent.start, end=sent.end, label=sent.label)
    # Add semantic similarity edges
    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_emb, norm_emb.T)
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            if sim_matrix[i, j] >= similarity_threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i, j]), type='semantic')
    # Add temporal adjacency edges
    for i in range(num_sentences - 1):
        G.add_edge(i, i+1, weight=temporal_decay, type='temporal')
    # Add construct label edges if available
    if construct_labels is not None and confidences is not None:
        for i in range(num_sentences):
            for j in range(i+1, num_sentences):
                if construct_labels[i] == construct_labels[j] and confidences[i] > construct_conf_threshold and confidences[j] > construct_conf_threshold:
                    G.add_edge(i, j, weight=min(confidences[i], confidences[j]), type='construct')
    return G

class SentenceGraphGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.gat_layers = torch.nn.ModuleList([
            GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h = torch.relu(self.input_proj(x))
        for gat, norm in zip(self.gat_layers, self.norms):
            h_res = h
            h = gat(h, edge_index)
            h = torch.nn.functional.dropout(h, p=0.1, training=self.training)
            if h.size(-1) == h_res.size(-1):
                h = h + h_res
            h = norm(h)
            h = torch.relu(h)
        return torch.nn.functional.log_softmax(self.classifier(h), dim=1)

def build_gnn_graph(sentences: List[Any], embeddings: np.ndarray, edge_threshold: float = 0.75) -> Data:
    """
    Build a PyTorch Geometric Data object for GNN training.
    """
    import numpy as np
    num_nodes = len(sentences)
    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_emb, norm_emb.T)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and sim_matrix[i, j] >= edge_threshold:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(embeddings, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data
