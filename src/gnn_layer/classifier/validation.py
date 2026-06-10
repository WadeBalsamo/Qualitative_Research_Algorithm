"""
gnn_layer/validation.py
-----------------------
The graph reliability gate — the over-smoothing safeguard and the trigger for
LLM-free scaling.

The GNN is a graph-distilled *student* of the multi-run LLM consensus (which is
itself bootstrapped by the human-validated subset). Before it may label NEW
segments on its own, it has to prove that it reproduces that consensus on
segments it did NOT train on. This module takes the cross-validated, out-of-sample
predictions from :func:`gnn_layer.train.crossval_predictions` and reports:

  * overall Cohen's κ of graph vs LLM-majority consensus (distillation fidelity),
  * per-VAAMR-stage and per-PURER-move support / recall / precision / binary κ —
    this per-class breakdown is what catches rare-but-critical Reappraisal
    erosion that an aggregate κ would hide,
  * κ(graph, human) and κ(LLM, human) on the 20% blind-coded subset (the
    independent quality axis),
  * an explicit "ready for LLM-free scaling?" verdict.

Outputs: 06_reports/report_gnn_validation.txt and 03_analysis_data/gnn/gnn_validation.csv.
"""

import csv
import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from process import output_paths as _paths
from .triangulation import _kappa

# Display names (match CLAUDE.md reference tables).
VAAMR_NAMES = {0: 'Vigilance', 1: 'Avoidance', 2: 'AttentionReg',
               3: 'Metacognition', 4: 'Reappraisal'}
PURER_NAMES = {0: 'P-Phenomenology', 1: 'U-Utilization', 2: 'R-Reframing',
               3: 'E-Educate/Expect', 4: 'R2-Reinforcement'}
# Rare-but-clinically-central stages whose recall we call out explicitly.
RARE_STAGE_FLOOR = 0.50
RARE_STAGES = (3, 4)  # Metacognition, Reappraisal
MIN_SUPPORT_FOR_FLOOR = 5


HELDOUT_PREDICTIONS_FILENAME = 'gnn_heldout_predictions.csv'


def write_heldout_predictions_csv(cv: dict, output_dir: str) -> str:
    """Persist the cross-validated, OUT-OF-FOLD GNN predictions per segment_id.

    These are predictions on segments the model did NOT train on (the example's
    own LLM label was withheld), so they are the honest, non-circular GNN axis for
    inter-rater-reliability comparison against the human codes. Written to
    ``03_analysis_data/gnn/gnn_heldout_predictions.csv``.

    ``cv`` is the dict returned by :func:`gnn_layer.train.crossval_predictions`;
    confidence is taken from ``*_logits`` (softmax max) when present.
    """
    import numpy as np

    def _conf_map(logit_key):
        out = {}
        for sid, logits in cv.get(logit_key, []) or []:
            arr = np.asarray(logits, dtype=float)
            e = np.exp(arr - arr.max())
            out[str(sid)] = float((e / e.sum()).max())
        return out

    v_conf = _conf_map('vaamr_logits')
    p_conf = _conf_map('purer_logits')
    rows: Dict[str, dict] = {}
    for sid, pred in (cv.get('vaamr') or []):
        rows.setdefault(str(sid), {})['vaamr_pred'] = int(pred)
        rows[str(sid)]['vaamr_conf'] = v_conf.get(str(sid))
    for sid, pred in (cv.get('purer') or []):
        rows.setdefault(str(sid), {})['purer_pred'] = int(pred)
        rows[str(sid)]['purer_conf'] = p_conf.get(str(sid))

    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, HELDOUT_PREDICTIONS_FILENAME)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['segment_id', 'gnn_vaamr_heldout_pred', 'gnn_vaamr_heldout_conf',
                    'gnn_purer_heldout_pred', 'gnn_purer_heldout_conf'])
        for sid in sorted(rows):
            r = rows[sid]
            w.writerow([sid, r.get('vaamr_pred', ''),
                        '' if r.get('vaamr_conf') is None else f"{r['vaamr_conf']:.4f}",
                        r.get('purer_pred', ''),
                        '' if r.get('purer_conf') is None else f"{r['purer_conf']:.4f}"])
    return path


def read_heldout_predictions(output_dir: str) -> Dict[str, dict]:
    """Read ``gnn_heldout_predictions.csv`` -> {segment_id: {vaamr_pred, vaamr_conf, ...}}.

    Returns ``{}`` if the file is absent (GNN not trained / no held-out preds).
    """
    path = os.path.join(_paths.gnn_data_dir(output_dir), HELDOUT_PREDICTIONS_FILENAME)
    if not os.path.isfile(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid = row['segment_id']
            def _i(v):
                return int(v) if v not in ('', None) else None
            def _f(v):
                return float(v) if v not in ('', None) else None
            out[sid] = {
                'vaamr_pred': _i(row.get('gnn_vaamr_heldout_pred')),
                'vaamr_conf': _f(row.get('gnn_vaamr_heldout_conf')),
                'purer_pred': _i(row.get('gnn_purer_heldout_pred')),
                'purer_conf': _f(row.get('gnn_purer_heldout_conf')),
            }
    return out


def _per_class(pairs: List[Tuple[int, int]], names: Dict[int, str]) -> List[dict]:
    """Per-class support/recall/precision/binary-κ from (pred, ref) integer pairs."""
    rows = []
    classes = sorted(set(r for _, r in pairs) | set(p for p, _ in pairs))
    for c in classes:
        support = sum(1 for _, r in pairs if r == c)
        tp = sum(1 for p, r in pairs if p == c and r == c)
        pred_c = sum(1 for p, _ in pairs if p == c)
        recall = (tp / support) if support else None
        precision = (tp / pred_c) if pred_c else None
        bin_k = _kappa([1 if p == c else 0 for p, _ in pairs],
                       [1 if r == c else 0 for _, r in pairs])
        rows.append({
            'class_id': c, 'class_name': names.get(c, str(c)),
            'support': support, 'recall': recall,
            'precision': precision, 'kappa': bin_k,
        })
    return rows


def _overall(pairs: List[Tuple[int, int]]) -> dict:
    if not pairs:
        return {'n': 0, 'percent_agreement': None, 'cohen_kappa': None}
    agree = sum(1 for p, r in pairs if p == r) / len(pairs)
    return {'n': len(pairs), 'percent_agreement': round(agree, 4),
            'cohen_kappa': _kappa([p for p, _ in pairs], [r for _, r in pairs])}


def evaluate_crossval(df_all, cv_preds: dict, config) -> dict:
    """Score out-of-sample predictions against LLM consensus (and human subset)."""
    import pandas as pd  # noqa: F401  (df_all is already a DataFrame)

    by_id = {str(r.get('segment_id')): r for _, r in df_all.iterrows()}

    def _ref(sid, key):
        r = by_id.get(str(sid))
        if r is None:
            return None
        v = r.get(key)
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    metrics: Dict[str, object] = {'irr_target': float(getattr(config, 'irr_target', 0.70))}

    # ---- VAAMR: graph vs LLM final_label ----
    v_pairs = [(int(pred), _ref(sid, 'final_label')) for sid, pred in cv_preds.get('vaamr', [])]
    v_pairs = [(p, r) for p, r in v_pairs if r is not None]
    metrics['vaamr_overall'] = _overall(v_pairs)
    metrics['vaamr_per_class'] = _per_class(v_pairs, VAAMR_NAMES)

    # ---- PURER: graph vs LLM purer_primary ----
    p_pairs = [(int(pred), _ref(sid, 'purer_primary')) for sid, pred in cv_preds.get('purer', [])]
    p_pairs = [(p, r) for p, r in p_pairs if r is not None]
    metrics['purer_overall'] = _overall(p_pairs)
    metrics['purer_per_class'] = _per_class(p_pairs, PURER_NAMES)

    # ---- human comparison (overall, on the blind-coded subset) ----
    def _human_axis(cv_key, ref_key):
        g, llm = [], []
        for sid, pred in cv_preds.get(cv_key, []):
            r = by_id.get(str(sid))
            if r is None or not bool(r.get('in_human_coded_subset', False)):
                continue
            h = r.get('human_label')
            try:
                h = int(h)
            except (ValueError, TypeError):
                continue
            g.append((int(pred), h))
            ll = _ref(sid, ref_key)
            if ll is not None:
                llm.append((ll, h))
        return {'graph_vs_human': _overall(g), 'llm_vs_human': _overall(llm)}

    metrics['vaamr_human'] = _human_axis('vaamr', 'final_label')

    # ---- scaling verdict ----
    v_k = metrics['vaamr_overall'].get('cohen_kappa')
    p_k = metrics['purer_overall'].get('cohen_kappa')
    tgt = metrics['irr_target']
    rare_ok = True
    rare_notes = []
    for row in metrics['vaamr_per_class']:
        if row['class_id'] in RARE_STAGES and row['support'] >= MIN_SUPPORT_FOR_FLOOR:
            rec = row['recall']
            if rec is not None and rec < RARE_STAGE_FLOOR:
                rare_ok = False
                rare_notes.append(
                    f"{row['class_name']} held-out recall {rec:.2f} < {RARE_STAGE_FLOOR:.2f} "
                    f"(n={row['support']}) — possible over-smoothing of a rare stage")
    vaamr_ready = (v_k is not None and v_k >= tgt and rare_ok)
    purer_ready = (p_k is not None and p_k >= tgt)
    metrics['rare_stage_notes'] = rare_notes
    metrics['vaamr_ready'] = bool(vaamr_ready)
    metrics['purer_ready'] = bool(purer_ready)
    metrics['ready_for_scaling'] = bool(vaamr_ready and (n_or_true(p_pairs, purer_ready)))
    return metrics


def n_or_true(p_pairs, purer_ready) -> bool:
    """PURER readiness only gates scaling when PURER held-out data exists."""
    return purer_ready if p_pairs else True


def _fmt(x, pct=False):
    if x is None:
        return '  n/a'
    return f"{x*100:5.1f}%" if pct else f"{x:+.3f}"


def write_validation_report(metrics: dict, output_dir: str, config=None) -> str:
    """Human-readable reliability-gate report → 06_reports/07_gnn/validation.txt."""
    W = 78
    tgt = metrics.get('irr_target', 0.70)
    L = []
    L.append("=" * W)
    L.append("GNN RELIABILITY GATE — OUT-OF-SAMPLE AGREEMENT WITH LLM CONSENSUS")
    L.append("=" * W)
    L.append("")
    L.append("The graph is a distilled student of the multi-run LLM majority-vote consensus.")
    L.append("Every number below is CROSS-VALIDATED: each segment was scored by a model that")
    L.append("did NOT train on it. Once the graph reproduces the LLM consensus to inter-rater")
    L.append(f"reliability (κ ≥ {tgt:.2f}) WITHOUT collapsing rare stages, it can label new")
    L.append("segments on its own — no LLM calls — and become the authoritative label of record.")
    L.append("")

    def _overall_block(title, ov, names_axis=None):
        L.append("-" * W)
        L.append(title)
        L.append("-" * W)
        if not ov or ov.get('n', 0) == 0:
            L.append("  (no out-of-sample predictions)")
            L.append("")
            return
        L.append(f"  n={ov['n']}   agreement={_fmt(ov['percent_agreement'], pct=True)}   "
                 f"κ={_fmt(ov['cohen_kappa'])}")
        L.append("")

    _overall_block("VAAMR (participant stages) — graph vs LLM consensus",
                   metrics.get('vaamr_overall'))

    def _per_class_table(rows):
        L.append(f"  {'Class':<18}{'n':>5}{'recall':>9}{'precision':>11}{'κ':>9}")
        L.append(f"  {'-'*18}{'-'*5:>5}{'-'*9:>9}{'-'*11:>11}{'-'*9:>9}")
        for r in rows:
            L.append(f"  {r['class_name']:<18}{r['support']:>5}"
                     f"{_fmt(r['recall'], pct=True):>9}{_fmt(r['precision'], pct=True):>11}"
                     f"{_fmt(r['kappa']):>9}")
        L.append("")

    if metrics.get('vaamr_per_class'):
        L.append("  Per-stage breakdown (recall on rare stages is the over-smoothing check):")
        _per_class_table(metrics['vaamr_per_class'])

    _overall_block("PURER (therapist moves) — graph vs LLM consensus",
                   metrics.get('purer_overall'))
    if metrics.get('purer_per_class'):
        _per_class_table(metrics['purer_per_class'])

    # Human axis
    hum = metrics.get('vaamr_human') or {}
    L.append("-" * W)
    L.append("Independent quality axis — agreement with the human blind-coded subset")
    L.append("-" * W)
    gvh, lvh = hum.get('graph_vs_human'), hum.get('llm_vs_human')
    if gvh and gvh.get('n'):
        L.append(f"  GRAPH vs HUMAN   n={gvh['n']:<5} κ={_fmt(gvh['cohen_kappa'])}")
        L.append(f"  LLM   vs HUMAN   n={lvh['n']:<5} κ={_fmt(lvh['cohen_kappa'])}")
        L.append("  (graph-vs-human approaching llm-vs-human ⇒ the student is as good as the teacher")
        L.append("   at matching human judgment, not merely echoing it.)")
    else:
        L.append("  (no human-coded subset yet — gate rests on graph-vs-LLM fidelity for now.)")
    L.append("")

    # Calibration (A3)
    cal = metrics.get('calibration')
    if cal:
        L.append("-" * W)
        L.append("Confidence calibration (A3) — temperature scaling on held-out logits")
        L.append("-" * W)
        _T = cal.get('temperature')
        L.append(f"  fitted temperature T = {_T:.3f} (T>1 softens over-confident probabilities)"
                 if isinstance(_T, (int, float)) else "  fitted temperature T = n/a")
        L.append(f"  ECE before = {_fmt(cal.get('ece_before'))}   "
                 f"after = {_fmt(cal.get('ece_after'))}   (n={cal.get('n')})")
        L.append("  Abstention floors (A2) and scale-mode labels use these calibrated probs.")
        L.append("")

    # Verdict
    L.append("=" * W)
    ready = metrics.get('ready_for_scaling')
    L.append(f"  READY FOR LLM-FREE SCALING?   {'YES' if ready else 'NO'}")
    L.append("=" * W)
    if metrics.get('rare_stage_notes'):
        for note in metrics['rare_stage_notes']:
            L.append(f"  ⚠ {note}")
    if ready:
        L.append("  The graph reproduces the LLM consensus to target reliability out-of-sample.")
        L.append("  You may FILL unlabeled data with the graph alone (TUI: 'Classify — Graph")
        L.append("  consensus', or `qra gnn classify`). NOTE: the graph is non-authoritative —")
        L.append("  it fills BELOW the LLM (tier gnn_consensus) and never overrides it. The probe")
        L.append("  is the recommended scaler (`qra probe`); see methodology §8.6.")
    else:
        L.append("  Keep classifying with the LLM consensus. Add more LLM/human-labeled segments")
        L.append("  and re-run the GNN layer until this gate reports YES.")
    L.append("")

    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'validation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def write_validation_csv(metrics: dict, output_dir: str) -> str:
    """Machine-readable per-class gate table → 03_analysis_data/gnn/gnn_validation.csv."""
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'gnn_validation.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['framework', 'scope', 'class_id', 'class_name',
                    'support', 'recall', 'precision', 'kappa'])
        for fw, key, ov_key in (('vaamr', 'vaamr_per_class', 'vaamr_overall'),
                                ('purer', 'purer_per_class', 'purer_overall')):
            ov = metrics.get(ov_key) or {}
            w.writerow([fw, 'overall', '', '', ov.get('n'), ov.get('percent_agreement'),
                        '', ov.get('cohen_kappa')])
            for r in metrics.get(key, []):
                w.writerow([fw, 'per_class', r['class_id'], r['class_name'],
                            r['support'], r['recall'], r['precision'], r['kappa']])
    return path


# ---------------------------------------------------------------------------
# Machine-readable gate verdict (Track 0.2 — gate-gated promotion)
# ---------------------------------------------------------------------------
# The whole safety of the bootstrapping chain rests on the reliability gate. A
# config flag (gnn_authoritative) must NOT be able to promote an un-gated graph to
# the label of record. So the gate persists its verdict here, and the orchestrator
# reads it before promotion: the graph becomes authoritative only when the operator
# opted in AND this verdict says ready_for_scaling. A missing/failing verdict forces
# no-promotion.

GATE_VERDICT_FILENAME = 'gnn_gate.json'


def write_gate_verdict(metrics: dict, output_dir: str) -> str:
    """Persist the machine-readable gate verdict → 03_analysis_data/gnn/gnn_gate.json."""
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, GATE_VERDICT_FILENAME)
    cal = metrics.get('calibration') or {}
    verdict = {
        'ready_for_scaling': bool(metrics.get('ready_for_scaling')),
        'vaamr_ready': bool(metrics.get('vaamr_ready')),
        'purer_ready': bool(metrics.get('purer_ready')),
        'vaamr_kappa': (metrics.get('vaamr_overall') or {}).get('cohen_kappa'),
        'purer_kappa': (metrics.get('purer_overall') or {}).get('cohen_kappa'),
        'rare_stage_ok': not bool(metrics.get('rare_stage_notes')),
        'rare_stage_notes': metrics.get('rare_stage_notes', []),
        'irr_target': metrics.get('irr_target'),
        # A3 calibration (None when not run) — scale-mode can reuse the fitted temperature.
        'calibration_temperature': cal.get('temperature'),
        'ece_before': cal.get('ece_before'),
        'ece_after': cal.get('ece_after'),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(verdict, f, indent=2)
    return path


def read_gate_verdict(output_dir: str) -> Optional[dict]:
    """Load the persisted gate verdict, or None if the gate has not run / is unreadable."""
    path = os.path.join(_paths.gnn_data_dir(output_dir), GATE_VERDICT_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def gate_ready_for_scaling(output_dir: str) -> bool:
    """True only if a persisted gate verdict exists AND reports ready_for_scaling.

    This is the hard precondition for GNN label-of-record promotion: a missing or
    failing verdict returns False, so an un-gated graph can never be authoritative.
    """
    v = read_gate_verdict(output_dir)
    return bool(v and v.get('ready_for_scaling'))


def format_gate_verdict(verdict, output_dir: str = '') -> str:
    """Render the reliability-gate verdict as a human-readable block.

    Single source of truth for the κ readout shared by ``qra gnn status`` (CLI)
    and the TUI's "View GNN validation details" action, so the two never drift.
    ``verdict`` is the dict from :func:`read_gate_verdict` (or None if the gate
    has not run).
    """
    lines = ["GNN reliability gate" + (f" — {output_dir}" if output_dir else "")]
    if not verdict:
        lines.append("  Status: not run (train the GNN layer first: `qra gnn train`).")
        return "\n".join(lines)

    ready = verdict.get('ready_for_scaling')
    status = "READY for LLM-free scaling" if ready else "NOT READY (keep LLM consensus)"
    target = verdict.get('irr_target')
    tgt = f"   (target κ ≥ {target})" if target is not None else ""

    def _row(name, kappa, ok):
        k = "n/a" if kappa is None else f"{kappa:.3f}"
        flag = "" if ok is None else ("  [ready]" if ok else "  [not ready]")
        return f"  {name} κ(graph,LLM): {k}{tgt}{flag}"

    lines.append(f"  Status:            {status}")
    lines.append(f"  Ready for scaling: {'YES' if ready else 'NO'}")
    lines.append(_row("VAAMR", verdict.get('vaamr_kappa'), verdict.get('vaamr_ready')))
    lines.append(_row("PURER", verdict.get('purer_kappa'), verdict.get('purer_ready')))
    notes = verdict.get('rare_stage_notes') or []
    lines.append(f"  Rare-stage checks: {'OK' if not notes else '; '.join(map(str, notes))}")
    if verdict.get('calibration_temperature') is not None:
        lines.append(f"  Calibration T:     {verdict['calibration_temperature']:.3f}")
    if verdict.get('timestamp'):
        lines.append(f"  Gate run at:       {verdict['timestamp']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scale-mode simulation gate (A5)
# ---------------------------------------------------------------------------
# The k-fold gate above trains on the same topology it scores. Real scaling attaches
# genuinely-NEW sessions inductively (kNN edges into the frozen training graph, no temporal
# context of their own). This simulates that: hold out whole sessions, attach them via
# attach_new_segments, and compare held-out κ to the in-sample CV κ. A large gap warns that
# the gate's κ over-states how well the graph will label brand-new sessions.

def scale_mode_simulation(df_all, segment_embeddings, config, framework=None,
                          vce_codes=None) -> dict:
    """Inductive whole-session holdout vs in-sample CV. Returns the κ gap + a risk flag."""
    import numpy as np
    from . import graph_builder as _gb
    from . import train as _tr
    from .inference import infer_head_predictions
    from ..soft_labels import build_soft_targets
    from .triangulation import _kappa

    if 'session_id' not in df_all.columns:
        return {'status': 'skipped: no session_id column'}
    df_all = df_all.copy()
    df_all['session_id'] = df_all['session_id'].astype(str)
    sessions = sorted(df_all['session_id'].unique())
    if len(sessions) < 2:
        return {'status': 'skipped: need >=2 sessions to simulate inductive attach'}

    vce_codes = list(vce_codes or [])
    n_vce = len(vce_codes)

    def _ref_int(sid):
        r = df_all.loc[df_all['segment_id'].astype(str) == str(sid), 'final_label']
        if r.empty:
            return None
        try:
            return int(r.iloc[0])
        except (ValueError, TypeError):
            return None

    hold_n = max(1, int(getattr(config, 'scale_sim_holdout_sessions', 1) or 1))
    rng = np.random.default_rng(int(config.seed))
    perm = list(rng.permutation(sessions))
    folds = [perm[i:i + hold_n] for i in range(0, len(perm), hold_n)]
    folds = [f for f in folds if len(sessions) - len(f) >= 1]

    sim_pairs = []
    n_attached = 0
    for hold_sessions in folds:
        hold_set = set(hold_sessions)
        train_df = df_all[~df_all['session_id'].isin(hold_set)]
        hold_df = df_all[df_all['session_id'].isin(hold_set)]
        train_ids = set(train_df['segment_id'].astype(str))
        train_emb = {sid: v for sid, v in segment_embeddings.items() if sid in train_ids}
        if not train_emb:
            continue
        g = _gb.build_graph(train_df, train_emb, config, framework=framework)
        soft = build_soft_targets(train_df, config.label_mode)
        tgts = _tr.assemble_targets(g, soft, config, df_all=train_df, vce_codes=vce_codes or None)
        model, _ = _tr.train_model(g, tgts, config, n_vce=n_vce)
        hold_ids = set(hold_df['segment_id'].astype(str))
        new_emb = {sid: v for sid, v in segment_embeddings.items()
                   if sid in hold_ids and sid not in g.index_of}
        if not new_emb:
            continue
        node_type_of = {}
        for _, r in hold_df.iterrows():
            spk = str(r.get('speaker', '') or '')
            node_type_of[str(r.get('segment_id'))] = (
                'participant_segment' if spk == 'participant'
                else 'therapist_segment' if spk == 'therapist' else 'segment')
        g2 = _gb.attach_new_segments(g, new_emb, config, node_type_of=node_type_of)
        hp = infer_head_predictions(model, g2, config)
        idmap = {str(sid): i for i, sid in enumerate(hp.get('segment_id', []))}
        vp = hp.get('gnn_vaamr_pred')
        if vp is None:
            continue
        for sid in new_emb:
            i = idmap.get(str(sid))
            if i is None or hp['node_type'][i] != 'participant_segment':
                continue
            r = _ref_int(sid)
            if r is None:
                continue
            sim_pairs.append((int(vp[i]), r))
            n_attached += 1

    kappa_sim = (_kappa([p for p, _ in sim_pairs], [r for _, r in sim_pairs])
                 if sim_pairs else None)

    # In-sample CV κ on the full graph (the gate's condition) for comparison.
    g_full = _gb.build_graph(df_all, segment_embeddings, config, framework=framework)
    soft_full = build_soft_targets(df_all, config.label_mode)
    tgts_full = _tr.assemble_targets(g_full, soft_full, config, df_all=df_all,
                                     vce_codes=vce_codes or None)
    cv = _tr.crossval_predictions(g_full, tgts_full, config, n_vce=n_vce)
    cv_pairs = [(int(p), _ref_int(sid)) for sid, p in cv.get('vaamr', [])]
    cv_pairs = [(p, r) for p, r in cv_pairs if r is not None]
    kappa_cv = (_kappa([p for p, _ in cv_pairs], [r for _, r in cv_pairs])
                if cv_pairs else None)

    gap = (kappa_cv - kappa_sim) if (kappa_cv is not None and kappa_sim is not None) else None
    max_gap = float(getattr(config, 'scale_sim_max_gap', 0.10))
    domain_shift_risk = bool(gap is not None and gap > max_gap)
    return {
        'n_sessions': len(sessions),
        'holdout_sessions_per_fold': hold_n,
        'n_attached_scored': n_attached,
        'kappa_cv_insample': kappa_cv,
        'kappa_inductive_holdout': kappa_sim,
        'gap': gap,
        'max_gap': max_gap,
        'domain_shift_risk': domain_shift_risk,
    }


def write_scale_sim_report(result: dict, output_dir: str) -> str:
    """Human-readable A5 scale-mode simulation report → 06_reports/07_gnn/."""
    def _k(x):
        return 'n/a' if not isinstance(x, (int, float)) else f"{x:+.3f}"

    W = 78
    L = ["=" * W, "SCALE-MODE SIMULATION GATE (A5)", "=" * W, ""]
    if result.get('status'):
        L.append(f"  {result['status']}")
        L.append("")
    else:
        L.append("The k-fold gate trains on the same topology it scores. Real scaling attaches")
        L.append("genuinely-new SESSIONS inductively (kNN into the frozen graph, no temporal")
        L.append("context of their own). This holds whole sessions out, attaches them, and")
        L.append("compares held-out κ to the in-sample CV κ. A large gap means the gate's κ")
        L.append("over-states how well the graph will label brand-new sessions.")
        L.append("")
        L.append(f"  sessions={result.get('n_sessions')}  "
                 f"held-out/fold={result.get('holdout_sessions_per_fold')}  "
                 f"attached+scored={result.get('n_attached_scored')}")
        L.append("")
        L.append(f"    κ in-sample (CV)        : {_k(result.get('kappa_cv_insample'))}")
        L.append(f"    κ inductive (new session): {_k(result.get('kappa_inductive_holdout'))}")
        L.append(f"    gap (CV − inductive)    : {_k(result.get('gap'))}   "
                 f"(flag if > {result.get('max_gap')})")
        L.append("")
        L.append("=" * W)
        L.append(f"  DOMAIN-SHIFT RISK: {'YES' if result.get('domain_shift_risk') else 'NO'}")
        L.append("=" * W)
        if result.get('domain_shift_risk'):
            L.append("  The graph labels brand-new sessions notably worse than the CV gate suggests.")
            L.append("  Prefer abstention/OOD deferral (A2/A3) and re-validate before LLM-free scaling.")
    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'scale_sim.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
