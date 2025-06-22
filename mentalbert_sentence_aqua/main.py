"""
main.py
-------

Entry point for the Mental-BERT AQUA pipeline. Orchestrates the full workflow:
- Loads configuration and data.
- Runs preprocessing, embedding, graph construction, clustering, classification, visualization, and evaluation in sequence.
- Supports command-line and programmatic execution for batch or interactive analysis.
- Designed for transparency, reproducibility, and extensibility in qualitative research workflows.
"""

import os
import torch
import logging
import networkx as nx
from community import community_louvain
from typing import List, Optional
from . import config, preprocessing, embedding, graph_construction, clustering, classification, visualization, audit_trail, evaluation

logging.basicConfig(level=logging.INFO)

def run_pipeline(srt_file: str, model_path: Optional[str] = None, lora_weights_path: Optional[str] = None, clustering_method: str = 'louvain') -> None:
    try:
        logging.info(f"Preprocessing file: {srt_file}")
        sentences = preprocessing.parse_srt_file(srt_file)
        sentences = preprocessing.segment_sentences(sentences)
        texts: List[str] = [preprocessing.normalize_text(s.text) for s in sentences]
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return
    try:
        logging.info("Embedding sentences...")
        embedder = embedding.MentalBERTEmbedder(model_path or config.FINETUNED_MODEL_PATH, lora_weights_path=lora_weights_path)
        embs = embedder.encode_sentences(texts)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return
    try:
        logging.info("Constructing GNN graph...")
        gnn_data = graph_construction.build_gnn_graph(sentences, embs.numpy(), edge_threshold=config.SIMILARITY_THRESHOLD)
    except Exception as e:
        logging.error(f"Graph construction failed: {e}")
        return
    if all(s.label for s in sentences):
        try:
            labels: List[int] = [config.CONSTRUCT_LABELS.index(s.label) for s in sentences]
            gnn_data.y = torch.tensor(labels, dtype=torch.long)
            gnn_model = graph_construction.SentenceGraphGNN(embs.shape[1], 64, len(config.CONSTRUCT_LABELS))
            optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
            gnn_model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                out = gnn_model(gnn_data)
                loss = torch.nn.functional.nll_loss(out, gnn_data.y)
                loss.backward()
                optimizer.step()
            torch.save(gnn_model.state_dict(), 'models/best_gnn.pt')
            logging.info("GNN model trained and saved.")
        except Exception as e:
            logging.error(f"GNN training failed: {e}")
            return
    try:
        gnn_model = graph_construction.SentenceGraphGNN(embs.shape[1], 64, len(config.CONSTRUCT_LABELS))
        gnn_model.load_state_dict(torch.load('models/best_gnn.pt', map_location=embedder.device))
        gnn_model.eval()
        with torch.no_grad():
            gnn_out = gnn_model(gnn_data)
            pred_labels = gnn_out.argmax(dim=1).cpu().numpy()
        logging.info("GNN inference complete.")
    except Exception as e:
        logging.error(f"GNN inference failed: {e}")
        return
    # 6. Community detection and label assignment
    # Compute confidences for construct label edges
    from mentalbert_sentence_aqua import classification, clustering as clust
    sent_labels: List[dict] = [
        {'label': s.label if s.label else '', 'confidence': 1.0} for s in sentences
    ]
    confidences = [l['confidence'] for l in sent_labels]
    construct_labels = [l['label'] for l in sent_labels]
    G: nx.Graph = graph_construction.build_sentence_graph(
        sentences, embs.numpy(), similarity_threshold=config.SIMILARITY_THRESHOLD,
        construct_labels=construct_labels, confidences=confidences
    )
    if clustering_method == 'louvain':
        partition: dict = community_louvain.best_partition(G)
    elif clustering_method == 'spectral':
        partition = clust.spectral_modularity_clustering(G, n_communities=len(config.CONSTRUCT_LABELS))
    elif clustering_method == 'greedy':
        partition = clust.greedy_modularity_communities(G, max_communities=len(config.CONSTRUCT_LABELS))
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    comm_labels: dict = classification.aggregate_community_labels(partition, sent_labels)
    visualization.plot_communities(G, partition, texts)
    audit = audit_trail.AuditTrail()
    audit.log_decision('sentence_labels', sent_labels)
    audit.log_decision('community_labels', comm_labels)
    # 7. Evaluation (if ground truth available)
    y_true = [s.label for s in sentences if s.label]
    y_pred = [l['label'] for l in sent_labels if l['label']]
    if y_true and y_pred and len(y_true) == len(y_pred):
        kappa = evaluation.compute_cohens_kappa(y_true, y_pred)
        prf = evaluation.compute_precision_recall_f1(y_true, y_pred, config.CONSTRUCT_LABELS)
        modularity = evaluation.compute_modularity(G, partition)
        audit.log_decision('evaluation', {
            'cohen_kappa': kappa,
            'precision_recall_f1': prf,
            'modularity': modularity
        })
        logging.info(f'Cohen kappa: {kappa:.3f}')
        logging.info(f'Modularity: {modularity:.3f}')
    else:
        logging.info('Ground truth labels not available for evaluation.')
    audit.save('audit_trail.json')
    logging.info('GNN-based AQUA pipeline complete.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Mental-BERT AQUA pipeline on a transcript.")
    parser.add_argument('--srt_file', type=str, required=True, help='Path to .srt transcript file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to fine-tuned Mental-BERT model')
    parser.add_argument('--lora_weights_path', type=str, default=None, help='Path to QLoRA weights')
    parser.add_argument('--clustering_method', type=str, default='louvain', choices=['louvain', 'spectral', 'greedy'], help='Clustering method to use')
    args = parser.parse_args()
    run_pipeline(args.srt_file, args.model_path, args.lora_weights_path, args.clustering_method)
