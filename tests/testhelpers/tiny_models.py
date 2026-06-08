"""
tests.testhelpers.tiny_models
------------------------------
Smallest real models + programmatic framework builders for the integration
tier (and any unit test that needs a ThemeFramework without the repo-root
``*_FRAMEWORK.md`` files, which are not present in every checkout).

TINY_EMBED   — all-MiniLM-L6-v2: a ~22M-param sentence-transformer, the
               smallest first-class embedding model. Used everywhere a real
               embedding is needed instead of the 8GB Qwen3 default.
build_tiny_config(output_dir) — a PipelineConfig wired for fast, hermetic-ish
               runs: tiny embeddings on all three embedding fields, n_runs=1,
               segmentation LLM refinement off, small GNN.
tiny_vaamr_framework() / tiny_purer_framework() — ThemeFramework objects built
               in code (no markdown), with the canonical 5 stages/moves.
"""
from __future__ import annotations

import os

TINY_EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def build_tiny_config(output_dir: str, transcript_dir: str = "", *,
                      enable_codebook: bool = False,
                      enable_gnn: bool = True):
    """Return a PipelineConfig sized for fast tests.

    All three embedding-model fields are set to TINY_EMBED so no large model
    downloads. LLM-backed segmentation refinement is disabled; n_runs=1.
    """
    from process.config import PipelineConfig

    cfg = PipelineConfig(
        transcript_dir=transcript_dir or os.path.join(output_dir, "input"),
        output_dir=output_dir,
    )
    # Tiny embeddings everywhere an embedding model is referenced. Guard each
    # sub-config with hasattr so the helper survives config-schema drift
    # (e.g. optional sub-configs that may not be present).
    for sub in ("segmentation", "codebook_embedding", "gnn_layer"):
        obj = getattr(cfg, sub, None)
        if obj is not None and hasattr(obj, "embedding_model"):
            obj.embedding_model = TINY_EMBED

    # Keep segmentation deterministic and offline.
    cfg.segmentation.use_llm_refinement = False
    cfg.segmentation.verbose_segmentation = False

    # Single-run classification (no multi-model IRR roster needed).
    for sub in ("theme_classification", "purer_classification"):
        obj = getattr(cfg, sub, None)
        if obj is not None and hasattr(obj, "n_runs"):
            obj.n_runs = 1

    # Feature toggles.
    cfg.run_codebook_classifier = enable_codebook
    cfg.gnn_layer.enabled = enable_gnn
    cfg.gnn_layer.epochs = 20
    cfg.gnn_layer.hidden_dim = 16
    cfg.gnn_layer.knn_k = 3
    cfg.gnn_layer.n_motif_clusters = 3
    cfg.gnn_layer.cache_embeddings = False
    cfg.gnn_layer.interpret_against_cf_ic = False
    cfg.gnn_layer.validation_folds = 2

    return cfg


# ---------------------------------------------------------------------------
# Programmatic frameworks (no markdown dependency)
# ---------------------------------------------------------------------------
def _theme(theme_id, key, name, short, alias_extra=None):
    from constructs.theme_schema import ThemeDefinition
    return ThemeDefinition(
        theme_id=theme_id,
        key=key,
        name=name,
        short_name=short,
        prompt_name=name,
        definition=f"Operational definition of {name}.",
        prototypical_features=[f"{name} feature A", f"{name} feature B"],
        distinguishing_criteria=f"What separates {name} from its neighbours.",
        exemplar_utterances=[f"An exemplar of {name}."],
        subtle_utterances=[f"A subtle case of {name}."],
        adversarial_utterances=[f"A near-miss for {name}."],
        word_prototypes=[name.lower()],
        aliases=alias_extra or [],
    )


def tiny_vaamr_framework():
    """A 5-stage VAAMR ThemeFramework built in code (matches canonical ids)."""
    from constructs.theme_schema import ThemeFramework
    themes = [
        _theme(0, "vigilance", "Vigilance", "VIG"),
        _theme(1, "avoidance", "Avoidance", "AVD"),
        _theme(2, "attention_regulation", "Attention Regulation", "ATT"),
        _theme(3, "metacognition", "Metacognition", "MET"),
        _theme(4, "reappraisal", "Reappraisal", "REA"),
    ]
    return ThemeFramework(name="VAAMR", version="test",
                          description="Test VAAMR framework", themes=themes)


def tiny_purer_framework():
    """A 5-move PURER ThemeFramework built in code (matches canonical ids)."""
    from constructs.theme_schema import ThemeFramework
    themes = [
        _theme(0, "phenomenological", "Phenomenological", "P"),
        _theme(1, "utilization", "Utilization", "U"),
        _theme(2, "reframing", "Reframing", "R"),
        _theme(3, "educate", "Educate/Expectancy", "E", alias_extra=["education", "educate"]),
        _theme(4, "reinforcement", "Reinforcement", "R2"),
    ]
    return ThemeFramework(name="PURER", version="test",
                          description="Test PURER framework", themes=themes)


def load_real_framework_or_skip(name: str):
    """Load the REAL framework from its repo-root ``*_FRAMEWORK.md`` via the
    registry, or ``raise unittest.SkipTest`` if the markdown is not present in
    this checkout.

    Methodology-conformance tests use this so they actually verify the canonical
    VAAMR/PURER content where the markdown exists (the maintainer's checkout),
    and skip cleanly in checkouts that omit the framework files.
    """
    import unittest
    try:
        from constructs.registry import load
        fw = load(name)
    except FileNotFoundError as e:
        raise unittest.SkipTest(f"{name} framework markdown not present: {e}")
    except KeyError as e:  # unknown framework name -> real test failure intent
        raise unittest.SkipTest(f"unknown framework {name!r}: {e}")
    if fw is None:
        raise unittest.SkipTest(f"framework {name!r} resolved to None")
    return fw
