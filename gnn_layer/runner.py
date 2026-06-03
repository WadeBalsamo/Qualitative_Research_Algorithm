"""
gnn_layer/runner.py
-------------------
Top-level entry point for the GNN representation-and-discovery layer (SCAFFOLD).

Invoked from analysis/runner.py ONLY when ``config.gnn_layer.enabled`` is True
(default False). Signature mirrors analysis.purer_analysis.run_purer_analysis so it
slots into the existing analysis orchestration.

Intended pipeline (each step is scaffolded in its own module):
    1. embeddings.load_or_build_segment_embeddings(df_all, cfg)        # Qwen3 reuse
    2. soft_labels.build_soft_targets(df_all, cfg.label_mode)          # Capability A target
    3. graph_builder.build_graph(df_all, embeddings, cfg, framework)
    4. train.train_model(graph, targets, cfg)  -> export_checkpoint
    5. inference.infer_segment_positions(...)  -> reports.write_segment_positions  # A
    6. inference.cue_block_embeddings(...) + motifs.*  -> reports.write_cue_motifs /
       write_emergent_motifs_report                                                # B
    7. gnn_lift.* + compare_gnn_vs_llm -> reports.write_gnn_vs_llm_lift            # C
    8. ablation.* (optional)                                                       # D
    9. coupling.* -> reports.write_coupling_report                                 # E

STATUS: scaffold. This function does NOT execute the (NotImplementedError) steps; it
logs that the layer is a scaffold and returns an empty result so enabling the layer
never breaks an analysis run. Replace the guarded early-return with the orchestration
above when implementing.
"""

from typing import Optional


def run_gnn_analysis(df_all, output_dir, framework=None, config=None, llm_client=None) -> dict:
    """Run the GNN layer over the assembled corpus DataFrame.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Full master_segments (all speakers), as built by analysis.loader.load_segments
        with speaker_filter=None.
    output_dir : str
        Pipeline output directory.
    framework : dict | ThemeFramework, optional
        VAAMR framework (for stage names / anchor definitions).
    config : GnnLayerConfig, optional
        The gnn_layer sub-config (PipelineConfig.gnn_layer).
    llm_client : optional
        Reused LLM client (unused by the GNN itself; accepted for signature parity).

    Returns
    -------
    dict : {'files_written': list, 'status': str, ...}
    """
    # ---- scaffold guard: never raise, never touch outputs ----
    print("  [gnn_layer] scaffold: run_gnn_analysis is not yet implemented — skipping. "
          "See docs/GNN_IMPLEMENTATION.md.")
    return {
        'files_written': [],
        'status': 'scaffold',
        'n_segments': (0 if df_all is None else len(df_all)),
    }
