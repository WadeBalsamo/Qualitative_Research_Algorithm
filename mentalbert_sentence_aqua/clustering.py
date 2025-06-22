"""
clustering.py
-------------

Performs community detection on the sentence graph using maximum modularity clustering (e.g., Louvain algorithm). Identifies clusters of semantically and thematically related sentences. Outputs cluster assignments for each sentence, supporting hierarchical and transparent theme discovery.
"""

import networkx as nx
from collections import defaultdict
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def louvain_community_detection(G: nx.Graph) -> dict:
    """
    Applies the Louvain algorithm to detect communities in the sentence graph.
    Returns a dict mapping node index to community id.
    """
    if community_louvain is None:
        raise ImportError("Please install the 'python-louvain' package for community detection.")
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def get_communities(partition: dict) -> dict:
    """
    Groups node indices by community id.
    """
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    return dict(communities)

def spectral_modularity_clustering(G: nx.Graph, n_communities: int = 5):
    """
    Spectral modularity maximization clustering (Newman method).
    """
    A = nx.adjacency_matrix(G)
    degrees = np.array(A.sum(axis=1)).flatten()
    m = A.sum() / 2.0
    degree_product = np.outer(degrees, degrees)
    B = A - degree_product / (2 * m)
    B = sp.csr_matrix(B)
    eigenvals, eigenvecs = eigsh(B, k=n_communities, which='LA')
    features = eigenvecs[:, :n_communities]
    kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
    community_labels = kmeans.fit_predict(features)
    return {i: int(community_labels[i]) for i in range(len(community_labels))}

def greedy_modularity_communities(G: nx.Graph, max_communities: int = 5):
    """
    Clauset-Newman-Moore greedy modularity maximization.
    Returns a list of sets of node indices.
    """
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G, weight='weight'))
    # Map node to community id
    partition = {}
    for cid, comm in enumerate(comms):
        for node in comm:
            partition[node] = cid
    return partition
