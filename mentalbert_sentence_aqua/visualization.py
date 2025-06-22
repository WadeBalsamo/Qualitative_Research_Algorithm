"""
visualization.py
----------------

Provides visualization tools for the AQUA pipeline:
- Graph community visualization (e.g., with NetworkX and matplotlib).
- Display of ranked similar sentences and their construct labels.
- Visual summaries of classification confidence and uncertainty.
- Supports both interactive and static output for qualitative review.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List

def plot_communities(G: nx.Graph, partition: Dict[int, int], sentences: List[str], figsize: tuple = (12, 8)) -> None:
    """
    Visualizes the sentence graph with nodes colored by community.
    Each node is labeled with a truncated sentence for interpretability.
    """
    pos = nx.spring_layout(G, seed=42)
    communities = set(partition.values())
    colors = plt.cm.rainbow([i / max(1, len(communities)-1) for i in range(len(communities))])
    color_map = {cid: colors[i] for i, cid in enumerate(communities)}
    node_colors = [color_map[partition[n]] for n in G.nodes()]
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels={i: s[:20]+'...' for i, s in enumerate(sentences)}, font_size=8)
    plt.title('Sentence Graph Communities')
    plt.axis('off')
    plt.show()

def show_top_similar_sentences(sim_matrix, sentences: List[str], idx: int, top_k: int = 5) -> None:
    """
    Displays the top-k most similar sentences to a given sentence index.
    """
    import numpy as np
    sims = sim_matrix[idx]
    top_indices = np.argsort(-sims)[1:top_k+1]
    print(f"\nQuery: {sentences[idx]}\nTop {top_k} similar sentences:")
    for i in top_indices:
        print(f"  [{i}] {sentences[i]} (sim={sims[i]:.2f})")
