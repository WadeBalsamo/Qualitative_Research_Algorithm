"""
gnn_layer/config.py
-------------------
Configuration dataclass for the GNN representation-and-discovery layer.

Placed here (mirroring ``codebook/config.py``) and imported into
``process/config.py`` as ``PipelineConfig.gnn_layer``. This is the one module in
the gnn_layer scaffold that is fully implemented — the layer is OFF by default
(``enabled=False``), so the pipeline and existing tests are unaffected until it is
explicitly turned on.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GnnLayerConfig:
    """Settings for the Qwen3-embedding GNN analysis layer.

    Defaults keep the layer OFF and, when enabled, reuse the same embedding model
    QRA already uses for segmentation and VCE coding so no second model is loaded.
    """
    # Master switch. When False, analysis/runner.py never invokes the layer.
    enabled: bool = False

    # ---- Embedding substrate (reuses codebook EmbeddingCodebookClassifier) ----
    embedding_model: str = 'Qwen/Qwen3-Embedding-8B'
    use_query_prefix: bool = True
    embedding_batch_size: int = 8
    cache_embeddings: bool = True          # cache to 02_meta/gnn/segment_embeddings.npz

    # ---- Graph construction ----
    knn_k: int = 8                         # kNN similarity edges per segment node
    include_vce_nodes: bool = True         # anchor nodes for VCE codes (ablatable via D)
    include_purer_nodes: bool = True
    include_microskill_nodes: bool = True
    cross_framework_min_lift: float = 1.5  # threshold for vaamr<->vce / purer<->microskill edges

    # ---- Model ----
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.5

    # ---- Objectives / training ----
    # Subset of: soft_vaamr, progression, vce_multilabel, purer, microskill_multilabel,
    #            contrastive, link_prediction
    objectives: List[str] = field(default_factory=lambda: [
        'soft_vaamr', 'progression', 'contrastive', 'link_prediction',
    ])
    # 'weak'  -> train soft heads on LLM multi-run ballots (default; directional)
    # 'human' -> train only on human-validated subset (for the independence claim)
    # 'self_supervised' -> link-prediction only, no LLM labels (independence control)
    label_mode: str = 'weak'
    contrastive_temp: float = 0.1
    epochs: int = 300
    lr: float = 1e-3
    patience: int = 40
    seed: int = 42
    device: Optional[str] = None           # None -> auto (cuda if available else cpu)

    # ---- Which speakers get GNN positioning ----
    run_on_participants: bool = True       # VAAMR mixture / progression coordinate
    run_on_therapists: bool = True         # cue-block embeddings / microskill positioning

    # ---- Capability B (motif discovery) ----
    n_motif_clusters: int = 12
    min_motif_influence: float = 1.2       # flag motifs whose forward-transition lift exceeds this
    motif_min_block_count: int = 3         # ignore motifs with fewer than this many cue blocks

    # ---- Capability E (coupling / latent factors) ----
    n_latent_factors: int = 5
    interpret_against_cf_ic: bool = True   # label latent factors against the inline CF/IC lexicon
