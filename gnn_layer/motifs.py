"""
gnn_layer/motifs.py
-------------------
Capability B — cue granularization + emergent-motif discovery (SCAFFOLD).

The flagship capability. Where analysis/purer_analysis.py collapses each cue block to
``dominant_purer`` (1 of 5) before measuring influence, this module keeps the
continuous cue-block embedding and:

  1. clusters cue-block embeddings into emergent therapist-language MOTIFS
     (finer than / orthogonal to the 5 PURER moves);
  2. regresses forward-transition outcome on the cue embedding, CONDITIONED on
     from_stage (preserving the stage-moderation hypothesis, methodology Sec 7.6),
     to score each motif's influence;
  3. flags high-influence / low-PURER-purity / low-microskill-purity motifs —
     influential therapist language that maps to no existing label — with exemplar
     cues for human review.

sklearn imported lazily.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .config import GnnLayerConfig


def cluster_cue_motifs(cue_embeddings: "np.ndarray", config: "GnnLayerConfig") -> "np.ndarray":
    """Cluster cue-block embeddings → integer motif id per block.

    TODO(scaffold): sklearn AgglomerativeClustering / KMeans with
    config.n_motif_clusters; return labels array.
    """
    raise NotImplementedError("gnn_layer.motifs.cluster_cue_motifs: scaffold")


def score_motif_influence(
    cue_embeddings: "np.ndarray",
    from_stages: "np.ndarray",
    forward_outcome: "np.ndarray",
    motif_ids: "np.ndarray",
    config: "GnnLayerConfig",
) -> Dict[int, dict]:
    """Per-motif influence on forward VAAMR transitions, conditioned on from_stage.

    Fits a from_stage-conditioned LogisticRegression of forward_outcome on the cue
    embedding and aggregates predicted lift per motif.

    Returns {motif_id: {influence, n_blocks, purer_purity, microskill_purity,
    dominant_purer, dominant_microskills}}.

    TODO(scaffold): sklearn LogisticRegression with from_stage one-hot; aggregate.
    """
    raise NotImplementedError("gnn_layer.motifs.score_motif_influence: scaffold")


def flag_emergent_motifs(motif_stats: Dict[int, dict], config: "GnnLayerConfig") -> List[int]:
    """Return motif ids that are influential but poorly explained by PURER/microskill.

    Criterion: influence >= config.min_motif_influence AND low label purity AND
    n_blocks >= config.motif_min_block_count.

    TODO(scaffold): filter motif_stats accordingly.
    """
    raise NotImplementedError("gnn_layer.motifs.flag_emergent_motifs: scaffold")


def select_motif_exemplars(motif_ids, cue_embeddings, df_blocks, k: int = 3) -> Dict[int, list]:
    """Pick representative exemplar cue texts per motif (nearest to centroid).

    TODO(scaffold): per motif, rank blocks by distance to centroid, take top-k texts.
    """
    raise NotImplementedError("gnn_layer.motifs.select_motif_exemplars: scaffold")
