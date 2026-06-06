"""
gnn_layer/runner.py
-------------------
Top-level entry point for the GNN representation-and-discovery layer.

Invoked from analysis/runner.py only when config.gnn_layer.enabled (ON by default).
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


def _independence_mode(config, df_all):
    """Resolve the label_mode for the G1 independence pass (LLM labels withheld).

    'human' / 'self_supervised' are honored verbatim. 'auto' prefers 'human' when a
    usable human blind-coded subset exists (>= independence_min_human rows), else falls
    back to the geometry-only 'self_supervised' NULL control. Returns None to skip.
    """
    m = getattr(config, 'independence_label_mode', 'auto')
    if m in ('human', 'self_supervised'):
        return m
    if m != 'auto':
        return None
    try:
        if 'in_human_coded_subset' in df_all.columns and 'human_label' in df_all.columns:
            n = int((df_all['in_human_coded_subset'] == True).sum())  # noqa: E712
            if n >= int(getattr(config, 'independence_min_human', 10)):
                return 'human'
    except Exception:
        pass
    return 'self_supervised'


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
        # Path B (G2): the MAIN graph is homogeneous by DEFAULT. Anchors are included
        # only when use_anchor_nodes is set (after the ablation justifies them on the
        # human axis) — never in the default substrate the triangulation relies on.
        anchor_features = anchor_edges = None
        if getattr(config, 'use_anchor_nodes', False):
            try:
                from . import anchors as _anc
                anchor_features, anchor_edges = _anc.build_anchors(df_all, seg_emb, config)
                if anchor_features:
                    _log(verbose, f"anchors ON: {len(anchor_features)} nodes, "
                                  f"{len(anchor_edges)} edges in main graph")
            except Exception as e:
                _log(verbose, f"anchor build failed ({e}); main graph stays homogeneous")
                anchor_features = anchor_edges = None
        graph = _gb.build_graph(df_all, seg_emb, config, framework=framework,
                                anchor_features=anchor_features, anchor_edges=anchor_edges)
        # Embeddings (segments + anchors) are done; free the ~8 GB Qwen3 from the GPU before
        # GNN training allocates, so the two do not coexist in VRAM. Downstream graph rebuilds
        # (ablations, scale-sim) reuse the cached seg_emb dict and never re-embed.
        _emb.release_embedder()
        soft = _sl.build_soft_targets(df_all, config.label_mode)
        # optional vocabulary for the vce head
        vce_codes = _vocabs(config)
        targets = _train.assemble_targets(graph, soft, config, df_all=df_all,
                                          vce_codes=vce_codes or None)
        model, metrics = _train.train_model(graph, targets, config, n_vce=len(vce_codes))
        _train.export_checkpoint(model, config, _paths.gnn_model_dir(output_dir), metrics)
        # Persist the trained graph so scale-mode can attach new segments inductively
        # (rather than rebuilding the graph from scratch). See run_gnn_classify.
        _gb.save_graph(graph, _paths.gnn_model_dir(output_dir))
        _log(verbose, f"trained GNN ({metrics.get('epochs_run')} epochs, "
                      f"best_loss={round(metrics.get('best_loss', 0.0), 4)})")
    except Exception as e:
        _log(verbose, f"graph/training failed: {e}")
        if verbose:
            traceback.print_exc()
        return {'files_written': files, 'status': f'failed: {e}', 'n_segments': len(df_all)}

    # ---- Reliability gate: out-of-sample per-stage / per-move κ vs LLM consensus ----
    # This is the over-smoothing safeguard and the trigger for LLM-free scaling.
    try:
        from . import validation as _val
        # Request held-out logits so A3 temperature calibration can reuse this CV.
        cv = _train.crossval_predictions(graph, targets, config, n_vce=len(vce_codes),
                                         return_logits=True)
        if cv['vaamr'] or cv['purer']:
            vm = _val.evaluate_crossval(df_all, cv, config)
            # ---- A3 temperature calibration (fit on the same held-out CV) ----
            if getattr(config, 'calibrate', False):
                try:
                    from . import calibration as _cal
                    cal = _cal.temperature_from_cv(cv, df_all)
                    config.calibration_temperature = cal['temperature']
                    vm['calibration'] = cal
                    _log(verbose, f"calibration: T={cal['temperature']:.3f}, "
                                  f"ECE {cal['ece_before']}→{cal['ece_after']}")
                except Exception as e:
                    _log(verbose, f"temperature calibration failed: {e}")
            files.append(_val.write_validation_report(vm, output_dir, config))
            files.append(_val.write_validation_csv(vm, output_dir))
            # Persist the machine-readable verdict so the orchestrator can gate
            # gnn_authoritative promotion on it (Track 0.2 — gate-gated promotion).
            files.append(_val.write_gate_verdict(vm, output_dir))
            _vk = (vm.get('vaamr_overall') or {}).get('cohen_kappa')
            _log(verbose, f"reliability gate: vaamr κ={_vk}, "
                          f"ready_for_scaling={vm.get('ready_for_scaling')}")
    except Exception as e:
        _log(verbose, f"reliability gate failed: {e}")
        if verbose:
            traceback.print_exc()

    # ---- Path-B anchor ablation (G2): do construct anchors earn their place? ----
    # Trains with/without anchors and scores Δκ on the GNN<->HUMAN axis (anchors inflate
    # GNN<->LLM by construction, so that axis is reported but never decisive).
    if getattr(config, 'run_anchor_ablation', False):
        try:
            from . import ablation as _abl
            anc_res = _abl.anchor_contribution(df_all, seg_emb, config, framework=framework)
            files.append(_abl.write_anchor_contribution_report(anc_res, output_dir))
            _log(verbose, f"anchor ablation: verdict={anc_res.get('verdict')}, "
                          f"recommend={anc_res.get('recommend_anchors')}")
        except Exception as e:
            _log(verbose, f"anchor ablation failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Track A1 checkpoint: do typed precipitates edges earn their place? ----
    # Builds the graph with/without the therapist->participant family on identical
    # folds/seed and compares out-of-sample κ on BOTH the human and LLM axes.
    if getattr(config, 'run_precipitates_ablation', False):
        try:
            from . import ablation as _abl
            prec_res = _abl.precipitates_contribution(df_all, seg_emb, config, framework=framework)
            files.append(_abl.write_precipitates_contribution_report(prec_res, output_dir))
            _log(verbose, f"precipitates ablation: verdict={prec_res.get('verdict')}, "
                          f"recommend={prec_res.get('recommend_precipitates')}")
        except Exception as e:
            _log(verbose, f"precipitates ablation failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- A4: semi-supervised label propagation (measured) ----
    if getattr(config, 'label_propagation', False):
        try:
            from . import propagation as _prop
            prop_res = _prop.propagation_contribution(graph, targets, config, df_all,
                                                      n_vce=len(vce_codes))
            files.append(_prop.write_propagation_report(prop_res, output_dir))
            _log(verbose, f"label propagation: verdict={prop_res.get('verdict')}, "
                          f"recommend={prop_res.get('recommend_propagation')}")
        except Exception as e:
            _log(verbose, f"label propagation failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- A5: scale-mode simulation gate (inductive whole-session holdout) ----
    if getattr(config, 'run_scale_sim', False):
        try:
            from . import validation as _val
            sim_res = _val.scale_mode_simulation(df_all, seg_emb, config,
                                                 framework=framework, vce_codes=vce_codes)
            files.append(_val.write_scale_sim_report(sim_res, output_dir))
            _log(verbose, f"scale-mode sim: κ_cv={sim_res.get('kappa_cv_insample')}, "
                          f"κ_inductive={sim_res.get('kappa_inductive_holdout')}, "
                          f"domain_shift_risk={sim_res.get('domain_shift_risk')}")
        except Exception as e:
            _log(verbose, f"scale-mode sim failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Capability A: per-segment positions ----
    positions = None
    try:
        from . import reports as _rep
        positions = _inf.infer_segment_positions(model, graph, config)
        files.append(_rep.write_segment_positions(positions, output_dir))
    except Exception as e:
        _log(verbose, f"positions failed: {e}")

    # ---- Capability C(i): extract trained head predictions + triangulate ----
    try:
        from . import reports as _rep
        from . import triangulation as _tri
        head_preds = _inf.infer_head_predictions(model, graph, config)
        if any(k.startswith('gnn_') for k in head_preds):
            files.append(_rep.write_gnn_head_predictions(head_preds, output_dir))
            tri = _tri.compute_triangulation(head_preds, df_all)
            if tri:
                files.append(_tri.write_triangulation_report(tri, output_dir))
                _log(verbose, "wrote GNN↔LLM↔human triangulation")
    except Exception as e:
        _log(verbose, f"head-prediction triangulation failed: {e}")
        if verbose:
            traceback.print_exc()

    # ---- Capability C(ii): INDEPENDENCE PASS (G1) — LLM labels withheld ----
    # The main model above trains on LLM ballots, so its GNN↔LLM agreement is
    # distillation fidelity. Here a SECOND model is trained with LLM labels withheld
    # ('human' supervises off the blind subset; 'self_supervised' is a geometry-only
    # NULL control) on the SAME graph, and its triangulation is written separately —
    # GNN↔LLM κ there is genuine corroboration. This is the substrate to cite for any
    # "independent measurement" claim.
    if getattr(config, 'report_independence_pass', True):
        try:
            from . import triangulation as _tri
            ind_mode = _independence_mode(config, df_all)
            if ind_mode:
                ind_soft = _sl.build_soft_targets(df_all, ind_mode)
                ind_targets = _train.assemble_targets(graph, ind_soft, config,
                                                      df_all=df_all, vce_codes=vce_codes or None)
                ind_model, _ = _train.train_model(graph, ind_targets, config, n_vce=len(vce_codes))
                ind_hp = _inf.infer_head_predictions(ind_model, graph, config)
                ind_tri = _tri.compute_triangulation(ind_hp, df_all)
                if ind_tri:
                    files.append(_tri.write_triangulation_report(
                        ind_tri, output_dir, mode=ind_mode,
                        filename='triangulation_independence.txt'))
                    _log(verbose, f"wrote independence-pass triangulation (mode={ind_mode})")
        except Exception as e:
            _log(verbose, f"independence pass failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Abstention floor calibration (A2): derive per-stage floors from held-out CV ----
    # When enabled, pick per-VAAMR-stage confidence floors that hit the target held-out
    # precision, then stash them on the config so infer_head_predictions abstains below them.
    if getattr(config, 'abstain_calibrate', False):
        try:
            cal = _train.calibrate_abstain_floors(graph, targets, config, df_all,
                                                  n_vce=len(vce_codes))
            config.abstain_per_stage = cal['floors']
            _log(verbose, f"calibrated abstention floors (target prec "
                          f"{cal['target_precision']}): {cal['floors']}")
        except Exception as e:
            _log(verbose, f"abstention calibration failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Consensus overlay: per-segment graph labels → 02_meta/classifications/gnn_labels.jsonl ----
    # Written every run (does NOT change labels-of-record unless gnn_authoritative=True;
    # see process/assembly/master_dataset.py). Raw LLM ballots stay visible per segment.
    if getattr(config, 'produce_consensus_labels', True):
        try:
            from process import classifications_io as _cio
            from classification_tools.data_structures import Segment as _Seg
            hp = _inf.infer_head_predictions(model, graph, config)
            sids = hp.get('segment_id', []); nts = hp.get('node_type', [])
            vp, vc = hp.get('gnn_vaamr_pred'), hp.get('gnn_vaamr_conf')
            va = hp.get('gnn_vaamr_abstain')
            pp, pc = hp.get('gnn_purer_pred'), hp.get('gnn_purer_conf')
            pa = hp.get('gnn_purer_abstain')
            csegs = []
            n_abstain = 0
            for i, sid in enumerate(sids):
                nt = nts[i]
                if nt not in ('participant_segment', 'therapist_segment'):
                    continue
                s = _Seg(segment_id=str(sid))
                if nt == 'participant_segment' and vp is not None:
                    s.gnn_vaamr_pred = int(vp[i]); s.gnn_vaamr_conf = float(vc[i])
                    if va is not None:
                        s.gnn_vaamr_abstain = bool(va[i]); n_abstain += int(va[i])
                if nt == 'therapist_segment' and pp is not None:
                    s.gnn_purer_pred = int(pp[i]); s.gnn_purer_conf = float(pc[i])
                    if pa is not None:
                        s.gnn_purer_abstain = bool(pa[i]); n_abstain += int(pa[i])
                s.gnn_label_source = 'gnn_trained'
                csegs.append(s)
            if csegs:
                files.append(_cio.merge_gnn_overlay(output_dir, csegs))
                _abmsg = f", {n_abstain} abstained" if (va is not None or pa is not None) else ""
                _log(verbose, f"wrote GNN consensus overlay ({len(csegs)} segments{_abmsg})")
        except Exception as e:
            _log(verbose, f"consensus overlay failed: {e}")
            if verbose:
                traceback.print_exc()

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
            # per-block PURER labels for purity
            ther_rows = {str(r.get('segment_id')): r for _, r in df_all.iterrows()
                         if str(r.get('speaker', '')) == 'therapist'}
            purer_lbl = []
            for r in rows:
                pls = [int(ther_rows[s]['purer_primary']) for s in r['therapist_seg_ids']
                       if s in ther_rows and _is_int(ther_rows[s].get('purer_primary'))]
                purer_lbl.append(Counter(pls).most_common(1)[0][0] if pls else None)
            purity = _mot.annotate_label_purity(motif_ids, purer_lbl)
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

    # ---- Capability C: GNN-vs-LLM lift ----
    try:
        from . import gnn_lift as _lift
        from . import reports as _rep
        if gnn_stage_by_id:
            gnn_t = _lift.gnn_vaamr_vce_lift(df_all, gnn_stage_by_id)
            llm_t = _lift.llm_vaamr_vce_lift(df_all)
            files.append(_rep.write_gnn_vs_llm_lift(_lift.compare_gnn_vs_llm(gnn_t, llm_t), output_dir))
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
            # Build each factor's exemplar THERAPIST TEXT so the CF/IC alliance naming
            # can actually run (previously None was passed → interpretation was a no-op).
            ther_text = {str(r.get('segment_id')): str(r.get('text', ''))
                         for _, r in df_all.iterrows()
                         if str(r.get('speaker', '')) == 'therapist'}
            scores = factors.get('block_scores')
            exemplar_texts_by_factor = {}
            if scores is not None:
                import numpy as np
                for f in range(scores.shape[1]):
                    top = np.argsort(scores[:, f])[::-1][:3]
                    texts = []
                    for i in top:
                        texts.append(' '.join(ther_text.get(s, '') for s in rows[i].get('therapist_seg_ids', [])).strip())
                    exemplar_texts_by_factor[int(f)] = [t for t in texts if t]
            interp = _cp.interpret_factors(factors, exemplar_texts_by_factor, config)
            files.append(_rep.write_coupling_factors(factors, exem, interp, output_dir))
            files.append(_rep.write_coupling_report(factors, exem, interp, output_dir))
    except Exception as e:
        _log(verbose, f"coupling failed: {e}")

    # ---- Capability D: ablation — which construct heads carry signal? ----
    if getattr(config, 'run_gnn_ablation', False):
        try:
            from . import ablation as _abl
            from . import reports as _rep
            abl_rows = []
            for head in ('vce', 'purer'):
                obj = {'vce': 'vce_multilabel', 'purer': 'purer'}[head]
                if obj not in config.objectives:
                    continue
                abl_rows.append(_abl.run_ablation(
                    graph, targets, config, ablate=head, n_vce=len(vce_codes)))
            if abl_rows:
                files.append(_rep.write_gnn_construct_signal(abl_rows, output_dir))
                _log(verbose, f"ablation: ranked {len(abl_rows)} construct heads by signal")
        except Exception as e:
            _log(verbose, f"ablation failed: {e}")

    # ---- VCE-on-VAAMR hypothesis: does the granular codebook sharpen the arc? ----
    # Direct A/B: held-out VAAMR κ with vs without the VCE multi-label layer (§3.3/§5.2).
    if getattr(config, 'test_vce_layer', False):
        try:
            from . import ablation as _abl
            from . import reports as _rep
            vce_res = _abl.vce_vaamr_contribution(graph, df_all, config)
            path = _rep.write_vce_contribution(vce_res, output_dir)
            if path:
                files.append(path)
                _log(verbose, f"VCE-on-VAAMR test: Δκ={vce_res.get('delta_kappa')} "
                              f"({vce_res.get('verdict')})")
            else:
                _log(verbose, f"VCE-on-VAAMR test {vce_res.get('status', 'produced no report')}")
        except Exception as e:
            _log(verbose, f"VCE-on-VAAMR test failed: {e}")

    # ---- Track B3/B4/B5: model-counterfactual influence (GATED on the reliability gate) ----
    # The observed Δprogression (analysis/mechanism.py) is the LEAD signal; this is the GNN's
    # secondary, corroborating lens. It runs ONLY from a gate-passing model — an un-gated
    # graph's counterfactuals are not trustworthy enough to inform the MindfulBERT dataset.
    if getattr(config, 'counterfactual', False):
        try:
            from . import validation as _val
            if not _val.gate_ready_for_scaling(output_dir):
                _log(verbose, "counterfactual influence suppressed: reliability gate not passed "
                              "(needs ready_for_scaling=YES)")
            else:
                from . import influence as _infl
                infl_res = _infl.counterfactual_influence(model, graph, df_all, config)
                if infl_res.get('status'):
                    _log(verbose, f"counterfactual influence {infl_res['status']}")
                else:
                    tri = _infl.triangulate(infl_res, output_dir)
                    p = _infl.write_influence_csv(infl_res, output_dir)
                    if p:
                        files.append(p)
                    files.append(_infl.write_influence_report(infl_res, tri, output_dir))
                    if getattr(config, 'counterfactual_subgroups', False):
                        sp = _infl.subgroup_influence(infl_res, df_all, output_dir)
                        if sp:
                            files.append(sp)
                    _log(verbose, f"counterfactual influence: {infl_res.get('n_blocks')} blocks"
                                  + (f", Spearman ρ={tri.get('spearman')}" if tri else ""))
        except Exception as e:
            _log(verbose, f"counterfactual influence failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Track D: subtext communities as routines (independent of the gate; discovery) ----
    # Uses the raw Qwen3 embeddings (recurring-language similarity), not the trained kNN graph.
    if getattr(config, 'subtext_communities', False):
        try:
            from . import communities as _comm
            comm_res = _comm.run_subtext_communities(df_all, seg_emb, output_dir, config)
            files.extend(comm_res.get('files_written', []))
            if comm_res.get('status') == 'ok':
                _log(verbose, f"subtext communities: {comm_res.get('n_communities')} found, "
                              f"{comm_res.get('n_stable')} stable (ARI={comm_res.get('ari')})")
            else:
                _log(verbose, f"subtext communities {comm_res.get('status')}")
        except Exception as e:
            _log(verbose, f"subtext communities failed: {e}")
            if verbose:
                traceback.print_exc()

    # ---- Figures (render whatever GNN data exists; each plotter is guarded) ----
    try:
        from . import figures as _gfigs
        fig_paths = _gfigs.generate_gnn_figures(output_dir, irr_target=getattr(config, 'irr_target', 0.70))
        files.extend(fig_paths)
        if fig_paths:
            _log(verbose, f"GNN figures: {[os.path.basename(p) for p in fig_paths]}")
    except Exception as e:
        _log(verbose, f"GNN figures failed: {e}")

    return {'files_written': files, 'status': 'ok', 'n_segments': len(df_all),
            'n_files': len(files)}


def _is_int(v):
    try:
        int(v)
        return True
    except (ValueError, TypeError):
        return False


def _vocabs(config):
    """Resolve the VCE code vocabulary the same way training does."""
    vce_codes: List[str] = []
    if 'vce_multilabel' in config.objectives:
        try:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            vce_codes = [c.code_id for c in get_phenomenology_codebook().codes]
        except Exception:
            vce_codes = []
    return vce_codes


def run_gnn_classify(df_all, output_dir, framework=None, config=None,
                     verbose=True, only_unlabeled=True) -> dict:
    """LLM-FREE classification of segments with the already-trained graph (scale mode).

    Loads the trained checkpoint and the persisted training graph (02_meta/gnn/),
    then attaches the unseen ``df_all`` segments INDUCTIVELY via kNN similarity edges
    into the frozen graph (GraphSAGE is inductive), runs a forward pass, and writes
    per-segment predictions to the gnn_labels overlay. No LLM calls, no training, no
    reports. Intended for adding new transcripts once the reliability gate
    (report_gnn_validation.txt) reports the graph is ready.

    Falls back to rebuilding the graph from ``df_all`` if no persisted graph exists
    (older checkpoints predate graph persistence).

    ``only_unlabeled=True`` writes overlay rows only for segments lacking an LLM label
    (the actual new data); set False to (re)label the whole corpus.
    """
    from .config import GnnLayerConfig
    if config is None:
        config = GnnLayerConfig(enabled=True)

    files: List[str] = []
    if df_all is None or len(df_all) == 0 or 'segment_id' not in df_all.columns:
        return {'files_written': files, 'status': 'skipped: empty/invalid dataframe', 'n_segments': 0}

    from process import output_paths as _paths
    from . import embeddings as _emb
    from . import graph_builder as _gb
    from . import train as _train
    from . import inference as _inf
    from process import classifications_io as _cio
    from classification_tools.data_structures import Segment as _Seg

    model_dir = _paths.gnn_model_dir(output_dir)
    weights = os.path.join(model_dir, 'weights.pt')
    if not os.path.isfile(weights):
        return {'files_written': files,
                'status': 'skipped: no trained checkpoint (run the GNN layer first)',
                'n_segments': len(df_all)}

    try:
        cache_path = os.path.join(model_dir, 'segment_embeddings.npz')
        seg_emb = _emb.load_or_build_segment_embeddings(df_all, config, cache_path=cache_path)
        vce_codes = _vocabs(config)
        base_graph = _gb.load_graph(model_dir)
        if base_graph is not None:
            # Inductive: attach only the unseen segments to the frozen training graph.
            new_emb = {sid: v for sid, v in seg_emb.items() if sid not in base_graph.index_of}
            node_type_of = {}
            for _, r in df_all.iterrows():
                sid = str(r.get('segment_id'))
                spk = str(r.get('speaker', '') or '')
                node_type_of[sid] = ('participant_segment' if spk == 'participant'
                                     else 'therapist_segment' if spk == 'therapist'
                                     else 'segment')
            graph = _gb.attach_new_segments(base_graph, new_emb, config,
                                            node_type_of=node_type_of) if new_emb else base_graph
        else:
            # Back-compat: no persisted graph → rebuild from df_all.
            _log(verbose, "no persisted training graph; rebuilding from df_all")
            graph = _gb.build_graph(df_all, seg_emb, config, framework=framework)
        model = _train.load_checkpoint(model_dir, graph, config, n_vce=len(vce_codes))
        # Reuse the temperature fitted at analyze time (A3) so scale-mode confidences are
        # calibrated identically to the gate; the persisted verdict is the source of truth.
        if getattr(config, 'calibrate', False) and getattr(config, 'calibration_temperature', None) is None:
            from . import validation as _val
            _v = _val.read_gate_verdict(output_dir) or {}
            if _v.get('calibration_temperature') is not None:
                config.calibration_temperature = float(_v['calibration_temperature'])
                _log(verbose, f"loaded calibration T={config.calibration_temperature:.3f} from gate verdict")
        hp = _inf.infer_head_predictions(model, graph, config)
    except Exception as e:
        _log(verbose, f"graph classify failed: {e}")
        if verbose:
            traceback.print_exc()
        return {'files_written': files, 'status': f'failed: {e}', 'n_segments': len(df_all)}

    # Which segment_ids already carry an LLM label (so we can target only the new ones).
    labeled_vaamr, labeled_purer = set(), set()
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id'))
        if _is_int(r.get('final_label')) or _is_int(r.get('primary_stage')):
            labeled_vaamr.add(sid)
        if _is_int(r.get('purer_primary')):
            labeled_purer.add(sid)

    # OOD gate (A3): a new segment far from the training support is an extrapolation, so
    # force ABSTAIN (defer to the LLM) regardless of softmax confidence.
    ood_over = set()
    _ood_thr = getattr(config, 'ood_threshold', None)
    if _ood_thr is not None and base_graph is not None:
        try:
            import numpy as np
            from . import calibration as _cal
            new_ids = [sid for sid in seg_emb if sid not in base_graph.index_of]
            if new_ids:
                base_X = base_graph.x.detach().cpu().numpy()
                new_X = np.stack([np.asarray(seg_emb[s], dtype='float32') for s in new_ids])
                scores = _cal.ood_scores(new_X, base_X, k=int(getattr(config, 'ood_knn_k', 8)))
                ood_over = {new_ids[i] for i in range(len(new_ids))
                            if float(scores[i]) > float(_ood_thr)}
                _log(verbose, f"OOD gate: {len(ood_over)}/{len(new_ids)} new segments "
                              f"beyond ood_threshold={_ood_thr}")
        except Exception as e:
            _log(verbose, f"OOD gate failed: {e}")

    sids, nts = hp.get('segment_id', []), hp.get('node_type', [])
    vp, vc = hp.get('gnn_vaamr_pred'), hp.get('gnn_vaamr_conf')
    va = hp.get('gnn_vaamr_abstain')
    pp, pc = hp.get('gnn_purer_pred'), hp.get('gnn_purer_conf')
    pa = hp.get('gnn_purer_abstain')
    csegs = []
    n_abstain = 0
    for i, sid in enumerate(sids):
        sid = str(sid); nt = nts[i]
        _is_ood = sid in ood_over
        if nt == 'participant_segment' and vp is not None:
            if only_unlabeled and sid in labeled_vaamr:
                continue
            s = _Seg(segment_id=sid)
            s.gnn_vaamr_pred = int(vp[i]); s.gnn_vaamr_conf = float(vc[i])
            _ab = bool(va[i]) if va is not None else False
            _ab = _ab or _is_ood
            if va is not None or _is_ood:
                s.gnn_vaamr_abstain = _ab; n_abstain += int(_ab)
            s.gnn_label_source = 'gnn_scale_mode'
            csegs.append(s)
        elif nt == 'therapist_segment' and pp is not None:
            if only_unlabeled and sid in labeled_purer:
                continue
            s = _Seg(segment_id=sid)
            s.gnn_purer_pred = int(pp[i]); s.gnn_purer_conf = float(pc[i])
            _ab = bool(pa[i]) if pa is not None else False
            _ab = _ab or _is_ood
            if pa is not None or _is_ood:
                s.gnn_purer_abstain = _ab; n_abstain += int(_ab)
            s.gnn_label_source = 'gnn_scale_mode'
            csegs.append(s)

    if csegs:
        files.append(_cio.merge_gnn_overlay(output_dir, csegs))
        _abmsg = f" ({n_abstain} abstained → LLM label kept)" if (va is not None or pa is not None) else ""
        _log(verbose, f"graph-classified {len(csegs)} segment(s) without LLMs{_abmsg}")
    return {'files_written': files, 'status': 'ok',
            'n_segments': len(df_all), 'n_classified': len(csegs)}
