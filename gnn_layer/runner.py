"""
gnn_layer/runner.py
-------------------
Top-level entry point for the GNN representation-and-discovery layer.

Invoked from analysis/runner.py only when config.gnn_layer.enabled (default False).
Signature mirrors analysis.purer_analysis.run_purer_analysis. Orchestrates:
  embeddings -> soft targets -> graph -> train -> inference (A) -> motifs (B) ->
  gnn lift (C) -> coupling (E). Each capability is wrapped so one failure does not
  abort the rest. Per-segment outputs go to 03_analysis_data/gnn/ (never master).
"""

import os
import traceback
from collections import Counter
from typing import Dict, List, Optional


def _log(verbose, msg):
    if verbose:
        print(f"  [gnn_layer] {msg}")


def run_gnn_analysis(df_all, output_dir, framework=None, config=None,
                     llm_client=None, verbose=True) -> dict:
    """Run the GNN layer over the assembled corpus DataFrame.

    Returns {'files_written': [...], 'status': str, ...}.
    """
    from .config import GnnLayerConfig
    if config is None:
        config = GnnLayerConfig(enabled=True)

    files: List[str] = []
    if df_all is None or len(df_all) == 0:
        return {'files_written': files, 'status': 'skipped: empty dataframe', 'n_segments': 0}
    if 'segment_id' not in df_all.columns or 'text' not in df_all.columns:
        return {'files_written': files, 'status': 'skipped: missing columns', 'n_segments': len(df_all)}

    from process import output_paths as _paths
    from . import embeddings as _emb
    from . import soft_labels as _sl
    from . import graph_builder as _gb
    from . import train as _train
    from . import inference as _inf

    # ---- embeddings (reuse Qwen3; cache to 02_meta/gnn) ----
    cache_path = os.path.join(_paths.gnn_model_dir(output_dir), 'segment_embeddings.npz')
    try:
        seg_emb = _emb.load_or_build_segment_embeddings(df_all, config, cache_path=cache_path)
    except Exception as e:
        _log(verbose, f"embeddings unavailable ({e}); skipping layer.")
        return {'files_written': files, 'status': 'skipped: embeddings unavailable',
                'n_segments': len(df_all)}
    if not seg_emb:
        return {'files_written': files, 'status': 'skipped: no embeddings', 'n_segments': len(df_all)}

    # ---- graph + soft targets + train ----
    try:
        graph = _gb.build_graph(df_all, seg_emb, config, framework=framework)
        soft = _sl.build_soft_targets(df_all, config.label_mode)
        # optional vocabularies for vce/microskill heads
        vce_codes, micro_codes = [], []
        if 'vce_multilabel' in config.objectives:
            try:
                from codebook.phenomenology_codebook import get_phenomenology_codebook
                vce_codes = [c.code_id for c in get_phenomenology_codebook().codes]
            except Exception:
                vce_codes = []
        if 'microskill_multilabel' in config.objectives:
            try:
                from codebook.microcounseling_codebook import get_microcounseling_codebook
                micro_codes = [c.code_id for c in get_microcounseling_codebook().codes]
            except Exception:
                micro_codes = []
        targets = _train.assemble_targets(graph, soft, config, df_all=df_all,
                                          vce_codes=vce_codes or None,
                                          micro_codes=micro_codes or None)
        model, metrics = _train.train_model(graph, targets, config,
                                            n_vce=len(vce_codes), n_microskill=len(micro_codes) or 8)
        _train.export_checkpoint(model, config, _paths.gnn_model_dir(output_dir), metrics)
        _log(verbose, f"trained GNN ({metrics.get('epochs_run')} epochs, "
                      f"best_loss={round(metrics.get('best_loss', 0.0), 4)})")
    except Exception as e:
        _log(verbose, f"graph/training failed: {e}")
        if verbose:
            traceback.print_exc()
        return {'files_written': files, 'status': f'failed: {e}', 'n_segments': len(df_all)}

    # ---- Capability A: per-segment positions ----
    positions = None
    try:
        from . import reports as _rep
        positions = _inf.infer_segment_positions(model, graph, config)
        files.append(_rep.write_segment_positions(positions, output_dir))
    except Exception as e:
        _log(verbose, f"positions failed: {e}")

    # segment_id -> gnn embedding row (for cue pooling); -> gnn stage (argmax mixture)
    seg_gnn_emb: Dict[str, "object"] = {}
    gnn_stage_by_id: Dict[str, int] = {}
    if positions is not None:
        import numpy as np
        emb_mat = positions['gnn_embedding']
        mix = positions.get('vaamr_mixture')
        for i, sid in enumerate(positions['segment_id']):
            seg_gnn_emb[sid] = emb_mat[i]
            if mix is not None and positions['node_type'][i] == 'participant_segment':
                gnn_stage_by_id[sid] = int(np.argmax(mix[i]))

    # ---- Capability B: cue motifs ----
    try:
        from . import motifs as _mot
        from . import reports as _rep
        blocks = _inf.build_cue_blocks_with_segments(df_all)
        rows, cue_X = _inf.cue_block_embeddings(blocks, seg_gnn_emb)
        if len(rows) >= 2:
            motif_ids = _mot.cluster_cue_motifs(cue_X, config)
            from_stages = [r['from_stage'] for r in rows]
            forward = [1 if r['to_stage'] > r['from_stage'] else 0 for r in rows]
            stats = _mot.score_motif_influence(cue_X, from_stages, forward, motif_ids, config)
            # per-block PURER + microskill labels for purity
            ther_rows = {str(r.get('segment_id')): r for _, r in df_all.iterrows()
                         if str(r.get('speaker', '')) == 'therapist'}
            purer_lbl, micro_lbl = [], []
            for r in rows:
                pls = [int(ther_rows[s]['purer_primary']) for s in r['therapist_seg_ids']
                       if s in ther_rows and _is_int(ther_rows[s].get('purer_primary'))]
                purer_lbl.append(Counter(pls).most_common(1)[0][0] if pls else None)
                msk = []
                for s in r['therapist_seg_ids']:
                    if s in ther_rows:
                        v = ther_rows[s].get('microskill_labels_ensemble')
                        if isinstance(v, list):
                            msk.extend(v)
                micro_lbl.append(msk)
            purity = _mot.annotate_label_purity(motif_ids, purer_lbl, micro_lbl)
            exem = _mot.select_motif_exemplars(motif_ids, cue_X, rows)
            flagged = _mot.flag_emergent_motifs(stats, purity, config)
            files.append(_rep.write_cue_motifs(stats, purity, exem, output_dir))
            # Per-block motif assignment so the mechanistic Δprogression analysis
            # can group cue blocks by emergent motif (additive sidecar).
            files.append(_rep.write_cue_block_assignments(rows, motif_ids, output_dir))
            files.append(_rep.write_emergent_motifs_report(flagged, stats, purity, exem, output_dir))
            _log(verbose, f"discovered {len(stats)} cue motifs ({len(flagged)} flagged emergent)")
    except Exception as e:
        _log(verbose, f"motif discovery failed: {e}")
        if verbose:
            traceback.print_exc()

    # ---- Capability C: GNN-vs-LLM lift + PURER x microskill ----
    try:
        from . import gnn_lift as _lift
        from . import reports as _rep
        if gnn_stage_by_id:
            gnn_t = _lift.gnn_vaamr_vce_lift(df_all, gnn_stage_by_id)
            llm_t = _lift.llm_vaamr_vce_lift(df_all)
            files.append(_rep.write_gnn_vs_llm_lift(_lift.compare_gnn_vs_llm(gnn_t, llm_t), output_dir))
        pm = _lift.purer_microskill_lift(df_all)
        if not pm.empty:
            files.append(_rep.write_purer_microskill_lift(pm, output_dir))
    except Exception as e:
        _log(verbose, f"lift tables failed: {e}")

    # ---- Capability E: coupling / latent factors ----
    try:
        from . import coupling as _cp
        from . import reports as _rep
        blocks = _inf.build_cue_blocks_with_segments(df_all)
        rows, cue_X = _inf.cue_block_embeddings(blocks, seg_gnn_emb)
        if len(rows) >= 2:
            forward = [1 if r['to_stage'] > r['from_stage'] else 0 for r in rows]
            factors = _cp.extract_latent_factors(cue_X, forward, config)
            exem = _cp.factor_exemplars(factors.get('block_scores'), rows)
            interp = _cp.interpret_factors(factors, None, config)
            files.append(_rep.write_coupling_factors(factors, exem, interp, output_dir))
            files.append(_rep.write_coupling_report(factors, exem, interp, output_dir))
    except Exception as e:
        _log(verbose, f"coupling failed: {e}")

    return {'files_written': files, 'status': 'ok', 'n_segments': len(df_all),
            'n_files': len(files)}


def _is_int(v):
    try:
        int(v)
        return True
    except (ValueError, TypeError):
        return False
