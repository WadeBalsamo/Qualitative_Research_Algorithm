"""
gnn_layer/train.py
------------------
Training loop + checkpoint export for the QRA GNN layer (SCAFFOLD).

Full-batch transductive training over the corpus graph with early stopping. Supports
the three label modes (weak LLM ballots / human-only / self-supervised) so the
independence claim (Capability C) can be made with a non-circular variant. Exports a
reproducible checkpoint bundle to ``02_meta/gnn/model/`` (weights.pt + manifest.json
with config, seed, n_nodes, data hash, metrics). Deterministic given config.seed.

torch imported lazily.
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_builder import HeteroGraph
    from .config import GnnLayerConfig


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs for reproducibility.

    TODO(scaffold): random.seed, np.random.seed, torch.manual_seed + deterministic flags.
    """
    raise NotImplementedError("gnn_layer.train.set_seed: scaffold")


def train_model(
    graph: "HeteroGraph",
    targets: dict,
    config: "GnnLayerConfig",
) -> Tuple[object, dict]:
    """Train the model; return (trained_module, metrics_dict).

    TODO(scaffold): build_model -> optimizer(Adam, lr) -> epoch loop with
    compute_losses, early stopping on a held-out split, restore best weights.
    """
    raise NotImplementedError("gnn_layer.train.train_model: scaffold")


def export_checkpoint(model, config: "GnnLayerConfig", model_dir: str,
                      metrics: Optional[dict] = None) -> str:
    """Persist weights.pt + manifest.json to ``model_dir`` (02_meta/gnn/model).

    TODO(scaffold): torch.save(state_dict); json.dump manifest(config, seed, metrics,
    data/git hash).
    """
    raise NotImplementedError("gnn_layer.train.export_checkpoint: scaffold")


def load_checkpoint(model_dir: str, graph: "HeteroGraph", config: "GnnLayerConfig"):
    """Rebuild the model and load weights from a checkpoint bundle.

    TODO(scaffold): read manifest, build_model(graph, config), load_state_dict.
    """
    raise NotImplementedError("gnn_layer.train.load_checkpoint: scaffold")
