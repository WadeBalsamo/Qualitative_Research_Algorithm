"""
experiments/gnn_reliability/harness.py
--------------------------------------
Measurement apparatus for the pre-registered GNN VAAMR reliability battery
(design_decisions.md §5 scoring, §6 arm battery).

This is the shared foundation every experiment arm plugs into. It owns:

  * ``load_corpus``    — master_segments.csv + the human-subset join (A0-pre).
  * ``build_folds``    — ONE participant-grouped + stratified fold map, shared by
                         every arm (identical folds/seed; §5 "folds").
  * ``get_embeddings`` — MiniLM (local cache) or Qwen3-8B (remote /v1/embeddings).
  * ``run_gnn_arm``    — the GNN measurement engine: build the graph once, then for
                         each grouped fold mask that fold's participant VAAMR targets,
                         train, and read out-of-fold predictions.
  * ``score_arm``      — the two reference axes (§5): LLM axis (vs final_label) and the
                         load-bearing human axis (vs human consensus, scored EXACTLY like
                         analysis/irr_analysis so the κ is comparable to 06b_irr_report.txt),
                         each with a participant-clustered bootstrap 95% CI; appends a
                         ledger row.
  * ``run_arm`` / ``__main__`` — dispatcher to run an arm end-to-end (arm "A0" wired).

κ math, the human-consensus read path, and the bootstrap are REUSED from the shipping
modules (analysis.irr_stats, process.irr_import, analysis.irr_analysis,
analysis.stats, gnn_layer.*) — nothing is re-derived here.

The shared config-flag contract (vaamr_n_classes, vaamr_class_balance,
vaamr_focal_gamma, vaamr_hard_ce_weight, precipitates_edges) is threaded through
``config`` to build_graph / train_model / build_soft_targets via ``getattr`` so it is
honored when those flags exist (another agent adds them) and is inert before then.
"""

import csv
import datetime
import os
import subprocess
import sys
from typing import Dict, List, Optional

# --- bootstrap src/ + repo root onto sys.path (run as a script or imported) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from process import output_paths as _paths  # noqa: E402
from process import irr_import  # noqa: E402
from analysis import irr_join  # noqa: E402
from analysis import irr_stats  # noqa: E402
from analysis import stats as _stats  # noqa: E402
from analysis.irr_analysis import _consensus_rows  # noqa: E402
from gnn_layer.classifier import validation as _val  # noqa: E402

DEFAULT_OUTPUT_DIR = 'data/Meta'

# 0–4 VAAMR + 5/-1 No-code; ledger recall column order (design CLAUDE.md stage table).
RECALL_COLUMNS = ['recall_vigilance', 'recall_avoidance', 'recall_attnreg',
                  'recall_metacog', 'recall_reappraisal', 'recall_nocode']

LEDGER_PATH = os.path.join(_ROOT, 'docs', 'gnn_experiments', 'ledger.csv')
LEDGER_COLUMNS = [
    'arm', 'branch', 'embedding', 'embed_dim', 'method', 'imbalance', 'n_classes',
    'gnn_human_kappa', 'gnn_human_lo', 'gnn_human_hi', 'gnn_human_n',
    'gnn_llm_kappa_205', 'gnn_llm_lo', 'gnn_llm_hi', 'gnn_llm_kappa_76',
    'recall_vigilance', 'recall_avoidance', 'recall_attnreg', 'recall_metacog',
    'recall_reappraisal', 'recall_nocode',
    'seed', 'timestamp', 'decision', 'notes',
]


# ---------------------------------------------------------------------------
# Corpus + human-subset join
# ---------------------------------------------------------------------------

def load_corpus(output_dir: str = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Read master_segments.csv and apply the human-subset join (A0-pre).

    Returns the full assembled DataFrame with ``in_human_coded_subset`` /
    ``human_label`` populated from the human consensus in ``qra.db``.
    """
    csv_path = os.path.join(_paths.master_segments_dir(output_dir), 'master_segments.csv')
    df_all = pd.read_csv(csv_path)
    df_all = irr_join.populate_human_columns(df_all, output_dir)
    return df_all


# ---------------------------------------------------------------------------
# Folds (built ONCE; shared by every arm)
# ---------------------------------------------------------------------------

def _labeled_participants(df_all: pd.DataFrame) -> pd.DataFrame:
    """Participant segments carrying an integer VAAMR ``final_label`` (the 205)."""
    part = df_all[df_all['speaker'] == 'participant'].copy()
    part = part[part['final_label'].notna()].copy()
    part['final_label'] = part['final_label'].astype(float).astype(int)
    part['segment_id'] = part['segment_id'].astype(str)
    part['participant_id'] = part['participant_id'].astype(str)
    return part


def build_folds(df_all: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                verbose: bool = True) -> Dict[str, int]:
    """Participant-grouped + stratified folds over the LABELED participant segments.

    ``sklearn.model_selection.StratifiedGroupKFold`` (groups=participant_id,
    y=final_label) so no participant is in train+test and the rare-stage mix is kept
    fold-to-fold. Built ONCE, deterministic given ``seed``. Returns
    ``{segment_id: fold_idx}`` over the 205 labeled segments.

    Logs fold structure and any fold that misses a rare class entirely (expected at
    n=205; design §5 "Honesty" — never hidden).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    lab = _labeled_participants(df_all)
    sids = lab['segment_id'].to_numpy()
    y = lab['final_label'].to_numpy()
    groups = lab['participant_id'].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of: Dict[str, int] = {}
    fold_rows: List[np.ndarray] = []
    for fi, (_train_idx, test_idx) in enumerate(sgkf.split(sids, y, groups)):
        fold_rows.append(test_idx)
        for i in test_idx:
            fold_of[str(sids[i])] = fi

    if verbose:
        names = _val.VAAMR_NAMES
        n_part = len(set(groups))
        print(f"[folds] {len(fold_of)} labeled participant segments, {n_part} participants, "
              f"{n_folds} folds (StratifiedGroupKFold, seed={seed})")
        for fi, test_idx in enumerate(fold_rows):
            dist = {int(k): int(v) for k, v in
                    zip(*np.unique(y[test_idx], return_counts=True))}
            parts = sorted(set(groups[test_idx]))
            missing = [names[c] for c in range(5) if dist.get(c, 0) == 0]
            print(f"  fold {fi}: n={len(test_idx):>3}  participants={len(parts)}  "
                  f"dist={ {names[c]: dist.get(c, 0) for c in range(5)} }"
                  + (f"  MISSING={missing}" if missing else ""))
        # cross-fold rare-class coverage (Avoidance, Metacognition are the binding gaps)
        for c in (1, 3):
            folds_with = sum(1 for ti in fold_rows if (y[ti] == c).sum() > 0)
            if folds_with < n_folds:
                print(f"  [rare-class gap] {names[c]} (label {c}) appears in only "
                      f"{folds_with}/{n_folds} folds")
    return fold_of


# ---------------------------------------------------------------------------
# Embeddings (MiniLM local cache / Qwen3-8B remote)
# ---------------------------------------------------------------------------

def get_embeddings(df_all: pd.DataFrame, which: str,
                   output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, np.ndarray]:
    """``{segment_id: vector}`` for every row of ``df_all`` with text.

    ``which='minilm'`` → existing local MiniLM-384 path, cached at
    ``02_meta/gnn/segment_embeddings.npz`` (a warm cache is a no-encode read).
    ``which='qwen'``   → remote OpenAI-compatible /v1/embeddings serving
    text-embedding-qwen3-embedding-8b (4096-d), cached at
    ``02_meta/gnn/segment_embeddings_qwen3_8b.npz``.

    Reuses ``gnn_layer.embeddings.load_or_build_segment_embeddings`` (text-hash keyed:
    only missing/stale rows are (re)encoded, so a warm cache never re-hits the model
    or the endpoint).
    """
    from gnn_layer.config import GnnLayerConfig
    from gnn_layer import embeddings as _emb

    gnn_dir = _paths.gnn_model_dir(output_dir)
    if which == 'minilm':
        config = GnnLayerConfig(
            embedding_backend='local',
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            cache_embeddings=True,
        )
        cache_path = os.path.join(gnn_dir, 'segment_embeddings.npz')
    elif which == 'qwen':
        config = GnnLayerConfig(
            embedding_backend='openai',
            embedding_base_url='http://10.0.0.58:1234/v1',
            embedding_model='text-embedding-qwen3-embedding-8b',
            use_query_prefix=True,
            embedding_batch_size=16,
            cache_embeddings=True,
        )
        cache_path = os.path.join(gnn_dir, 'segment_embeddings_qwen3_8b.npz')
    else:
        raise ValueError(f"get_embeddings: unknown embedding '{which}' (use 'minilm' | 'qwen')")

    return _emb.load_or_build_segment_embeddings(df_all, config, cache_path=cache_path)


# ---------------------------------------------------------------------------
# GNN measurement engine (grouped-CV out-of-fold predictions)
# ---------------------------------------------------------------------------

def run_gnn_arm(df_all: pd.DataFrame, embeddings: Dict[str, np.ndarray],
                folds: Dict[str, int], config) -> Dict[str, int]:
    """Out-of-fold VAAMR predictions via the shared participant-grouped folds.

    Build the graph ONCE; for each fold, mask that fold's participant VAAMR soft
    targets, train a fresh model on the rest, and read the held-out argmax. Because
    the folds are participant-grouped, ALL of a held-out participant's segments
    (labeled AND "No code") are out-of-fold together, so every human-coded item is
    naturally out-of-fold (design §5). No-code participants not covered by ``folds``
    (a pure-No-code participant) are assigned to a fold by participant deterministically.

    Returns ``{segment_id: pred_int}`` (argmax in ``0..n_classes-1``) for every
    participant segment carrying a VAAMR soft target — a superset of the labeled 205,
    so the human axis can score the full usable consensus.

    NOT ``train.crossval_predictions`` (that is random k-fold). The full ``config`` is
    threaded into build_graph / build_soft_targets / train_model so downstream
    loss/head/edge flags (vaamr_n_classes, class balance, focal, precipitates_edges …)
    are honored when present.
    """
    import torch
    import torch.nn.functional as F
    from gnn_layer.classifier import graph_builder as _gb
    from gnn_layer import soft_labels as _sl
    from gnn_layer.classifier import train as _train
    from gnn_layer.runner import _vocabs

    n_classes = int(getattr(config, 'vaamr_n_classes', 5))

    graph = _gb.build_graph(df_all, embeddings, config, framework=None)
    soft = _sl.build_soft_targets(df_all, config.label_mode, n_stages=n_classes)
    vce_codes = _vocabs(config)
    targets = _train.assemble_targets(graph, soft, config, df_all=df_all,
                                      vce_codes=vce_codes or None)

    v_idx = targets.get('vaamr_idx')
    n_v = int(v_idx.numel()) if v_idx is not None else 0
    if n_v == 0:
        return {}
    node_ids = list(graph.node_ids)
    pos_sid = [str(node_ids[int(v_idx[i])]) for i in range(n_v)]

    # segment_id -> participant_id (for grouping the No-code segments by their participant)
    part_of = {str(r.get('segment_id')): str(r.get('participant_id'))
               for _, r in df_all.iterrows()}

    # participant -> fold from the (labeled) fold map; all of a participant's labeled
    # segments share a fold by construction, so first-seen is consistent.
    n_folds = (max(folds.values()) + 1) if folds else 1
    pfold: Dict[str, int] = {}
    for sid, f in folds.items():
        p = part_of.get(str(sid))
        if p is not None:
            pfold.setdefault(p, int(f))
    # Assign participants with NO labeled segment (pure No-code) deterministically.
    extra = sorted({part_of.get(s) for s in pos_sid} - set(pfold) - {None})
    for i, p in enumerate(extra):
        pfold[p] = i % n_folds

    pos_fold = [pfold.get(part_of.get(pos_sid[i])) for i in range(n_v)]

    p_idx = targets.get('purer_idx')
    n_p = int(p_idx.numel()) if p_idx is not None else 0
    keep_p = list(range(n_p))  # PURER is auxiliary here — never masked

    oof: Dict[str, int] = {}
    for f in range(n_folds):
        held = [i for i in range(n_v) if pos_fold[i] == f]
        if not held:
            continue
        keep_v = [i for i in range(n_v) if pos_fold[i] != f]
        sub = _train._subset_targets(targets, keep_v, keep_p)
        fold_cfg = _train._replace_seed(config, int(config.seed) + 1 + f)
        model, _ = _train.train_model(graph, sub, fold_cfg, n_vce=len(vce_codes))
        dev = _train._device(fold_cfg)
        x = graph.x.to(dev)
        ei = graph.edge_index.to(dev)
        ew = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
        eti = (graph.edge_type_ids.to(dev)
               if getattr(graph, 'edge_type_ids', None) is not None else None)
        model.eval()
        with torch.no_grad():
            res = model(x, ei, ew, eti)
        probs = F.softmax(res['soft_vaamr'], dim=1).cpu().numpy()
        for pos in held:
            gi = int(v_idx[pos])
            oof[pos_sid[pos]] = int(probs[gi].argmax())
    return oof


# ---------------------------------------------------------------------------
# Scoring (LLM axis + load-bearing human axis) + ledger row
# ---------------------------------------------------------------------------

def _kappa_cluster_ci(a: List[int], b: List[int], clusters: List,
                      seed: int = 42, n_boot: int = 2000) -> dict:
    """Participant-clustered bootstrap 95% CI for Cohen's κ between aligned label
    lists ``a`` / ``b``, clustered by ``clusters`` (participant_id).

    Reuses ``analysis.stats.cluster_bootstrap_ci`` — each item's (a, b) pair is packed
    into one finite float so resampling WHOLE participants preserves the pairing; the
    statistic unpacks and computes κ via ``irr_stats.cohen_kappa``. ``point`` equals the
    plain κ over all items.
    """
    if len(a) < 2:
        return {'point': irr_stats.cohen_kappa(a, b), 'lo': None, 'hi': None,
                'n': len(a), 'n_clusters': len(set(clusters))}
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    # labels live in [-1, 5] -> +1 into [0, 6]; pack as (a+1)*10 + (b+1) in [0, 66].
    packed = ((a_arr + 1) * 10 + (b_arr + 1)).astype(float)

    def _stat(arr: np.ndarray) -> float:
        codes = arr.astype(int)
        aa = (codes // 10) - 1
        bb = (codes % 10) - 1
        k = irr_stats.cohen_kappa(aa.tolist(), bb.tolist())
        return float('nan') if k is None else float(k)

    res = _stats.cluster_bootstrap_ci(packed, list(clusters), statistic=_stat,
                                      n_boot=n_boot, seed=seed)
    # point from the helper is κ over all items; keep the exact κ for the headline.
    res['point'] = irr_stats.cohen_kappa(a, b)
    return res


def score_arm(arm: str, oof_preds: Dict[str, int], df_all: pd.DataFrame,
              output_dir: str, n_classes: int, meta: Optional[dict] = None,
              write_ledger: bool = True) -> dict:
    """Score one arm's out-of-fold predictions on BOTH reference axes (§5) and append
    a ``docs/gnn_experiments/ledger.csv`` row. Returns the full metrics dict.

    LLM axis   — oof vs ``final_label`` over the 205 (+ per-class recall/precision/
                 binary-κ mirroring ``gnn_layer.validation._per_class``; + κ over the
                 human-coded subset; + participant-clustered bootstrap 95% CI).
    Human axis — oof vs the human consensus, scored EXACTLY like ``analysis.irr_analysis``
                 (reuse irr_import.read_human_codes + _consensus_rows + cohen_kappa;
                 "No code"=-1 is a 6th category; a 6-class arm maps predicted class 5 → -1)
                 so the κ is directly comparable to ``06b_irr_report.txt``; + CI.
    """
    meta = dict(meta or {})
    oof_preds = {str(k): int(v) for k, v in oof_preds.items()}
    master_sids = set(df_all['segment_id'].astype(str))
    part_of = {str(r.get('segment_id')): str(r.get('participant_id'))
               for _, r in df_all.iterrows()}

    def _map_pred(p):
        """6-class arms map predicted No-code (class 5) → ABSTAIN(-1) for the human axis."""
        return -1 if (n_classes >= 6 and p == 5) else p

    # ---- LLM axis: oof vs final_label over the labeled 205 ----
    lab = _labeled_participants(df_all)
    final_of = {str(r['segment_id']): int(r['final_label']) for _, r in lab.iterrows()}
    llm_preds, llm_refs, llm_groups = [], [], []
    for sid, ref in final_of.items():
        if sid in oof_preds:
            llm_preds.append(oof_preds[sid])
            llm_refs.append(ref)
            llm_groups.append(part_of.get(sid))
    llm_ci = _kappa_cluster_ci(llm_preds, llm_refs, llm_groups)
    per_class = _val._per_class(list(zip(llm_preds, llm_refs)), _val.VAAMR_NAMES)
    recall_by_class = {r['class_id']: r['recall'] for r in per_class}

    # LLM axis restricted to the human-coded subset (the "76": every consensus item).
    codes = irr_import.read_human_codes(output_dir)
    human_sids = {str(c['segment_id']) for c in codes
                  if c.get('is_consensus') and c.get('segment_id')}
    sub_preds, sub_refs = [], []
    for sid in human_sids:
        if sid in final_of and sid in oof_preds:
            sub_preds.append(oof_preds[sid])
            sub_refs.append(final_of[sid])
    llm_kappa_76 = irr_stats.cohen_kappa(sub_preds, sub_refs)

    # ---- Human axis (load-bearing): oof vs human consensus, irr_analysis semantics ----
    consensus = _consensus_rows(codes)
    h_list, m_list, h_groups = [], [], []
    n_deferred = n_excluded = 0
    for c in consensus:
        sid = c.get('segment_id')
        if not sid or str(sid) not in master_sids:   # seg must resolve in master
            n_excluded += 1
            continue
        sid = str(sid)
        if sid not in oof_preds:                      # GNN deferred (no out-of-fold pred)
            n_deferred += 1
            continue
        h = c.get('primary')
        if h is None:
            continue
        h_list.append(int(h))
        m_list.append(_map_pred(oof_preds[sid]))
        h_groups.append(part_of.get(sid))
    human_ci = _kappa_cluster_ci(m_list, h_list, h_groups)

    result = {
        'arm': arm,
        'n_classes': n_classes,
        'llm_axis': {
            'cohen_kappa_205': llm_ci['point'],
            'ci95': [llm_ci['lo'], llm_ci['hi']],
            'n': llm_ci['n'],
            'n_clusters': llm_ci['n_clusters'],
            'cohen_kappa_76': llm_kappa_76,
            'n_76': len(sub_preds),
            'per_class': per_class,
        },
        'human_axis': {
            'cohen_kappa': human_ci['point'],
            'ci95': [human_ci['lo'], human_ci['hi']],
            'n': human_ci['n'],
            'n_clusters': human_ci['n_clusters'],
            'n_deferred': n_deferred,
            'n_excluded': n_excluded,
        },
        'meta': meta,
    }
    if write_ledger:
        _append_ledger_row(result, meta)
    return result


def _git_branch() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                             cwd=_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _append_ledger_row(result: dict, meta: dict) -> str:
    """Append one row to docs/gnn_experiments/ledger.csv (header already exists)."""
    llm, hum = result['llm_axis'], result['human_axis']
    recalls = {r['class_id']: r['recall'] for r in llm['per_class']}
    row = {
        'arm': result['arm'],
        'branch': meta.get('branch') or _git_branch(),
        'embedding': meta.get('embedding', ''),
        'embed_dim': meta.get('embed_dim', ''),
        'method': meta.get('method', ''),
        'imbalance': meta.get('imbalance', ''),
        'n_classes': result['n_classes'],
        'gnn_human_kappa': _fmt(hum['cohen_kappa']),
        'gnn_human_lo': _fmt(hum['ci95'][0]),
        'gnn_human_hi': _fmt(hum['ci95'][1]),
        'gnn_human_n': hum['n'],
        'gnn_llm_kappa_205': _fmt(llm['cohen_kappa_205']),
        'gnn_llm_lo': _fmt(llm['ci95'][0]),
        'gnn_llm_hi': _fmt(llm['ci95'][1]),
        'gnn_llm_kappa_76': _fmt(llm['cohen_kappa_76']),
        'seed': meta.get('seed', ''),
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'decision': meta.get('decision', ''),
        'notes': meta.get('notes', ''),
    }
    for i, col in enumerate(RECALL_COLUMNS):
        row[col] = _fmt(recalls.get(i))

    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    need_header = (not os.path.isfile(LEDGER_PATH)) or os.path.getsize(LEDGER_PATH) == 0
    with open(LEDGER_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction='ignore')
        if need_header:
            w.writeheader()
        w.writerow(row)
    return LEDGER_PATH


# ---------------------------------------------------------------------------
# Arm dispatcher
# ---------------------------------------------------------------------------

def _arm_spec(arm_name: str):
    """Return (embedding, config, method, imbalance, n_classes) for a pre-registered arm.

    Only A0 (the harness self-check baseline) is wired here; the other §6 arms are
    added by the battery agent. Each sets the shared config flags; ``run_gnn_arm``
    threads them through, so adding an arm is a spec entry, not a harness change.
    """
    from gnn_layer.config import GnnLayerConfig
    if arm_name == 'A0':
        return dict(embedding='minilm', config=GnnLayerConfig(),
                    method='GraphSAGE', imbalance='none', n_classes=5)
    raise ValueError(f"run_arm: arm '{arm_name}' not wired in this harness "
                     f"(A0 only; see design_decisions.md §6 for the battery)")


def run_arm(arm_name: str, output_dir: str = DEFAULT_OUTPUT_DIR,
            verbose: bool = True) -> dict:
    """End-to-end: load corpus → build folds → embed → run GNN arm → score + ledger."""
    spec = _arm_spec(arm_name)
    df_all = load_corpus(output_dir)
    folds = build_folds(df_all, seed=int(spec['config'].seed), verbose=verbose)
    emb = get_embeddings(df_all, spec['embedding'], output_dir)
    embed_dim = int(len(next(iter(emb.values())))) if emb else None
    if verbose:
        print(f"[{arm_name}] embedding={spec['embedding']} dim={embed_dim} "
              f"nodes={len(emb)}  training grouped-CV ...")
    oof = run_gnn_arm(df_all, emb, folds, spec['config'])
    meta = {
        'embedding': spec['embedding'], 'embed_dim': embed_dim,
        'method': spec['method'], 'imbalance': spec['imbalance'],
        'seed': int(spec['config'].seed), 'branch': _git_branch(),
    }
    result = score_arm(arm_name, oof, df_all, output_dir, spec['n_classes'], meta=meta)
    if verbose:
        _print_result(result)
    return result


def _print_result(result: dict) -> None:
    llm, hum = result['llm_axis'], result['human_axis']
    print("=" * 72)
    print(f"ARM {result['arm']}  (n_classes={result['n_classes']})")
    print("-" * 72)
    print(f"  LLM axis   κ(205) = {_fmt(llm['cohen_kappa_205'])}  "
          f"[{_fmt(llm['ci95'][0])}, {_fmt(llm['ci95'][1])}]  "
          f"n={llm['n']} ({llm['n_clusters']} participants)")
    print(f"             κ(human-coded subset) = {_fmt(llm['cohen_kappa_76'])}  n={llm['n_76']}")
    print("             per-class recall: " + ", ".join(
        f"{r['class_name']}={_fmt(r['recall'])}" for r in llm['per_class'] if r['class_id'] < 5))
    print(f"  HUMAN axis κ      = {_fmt(hum['cohen_kappa'])}  "
          f"[{_fmt(hum['ci95'][0])}, {_fmt(hum['ci95'][1])}]  "
          f"n={hum['n']} ({hum['n_clusters']} participants)  "
          f"deferred={hum['n_deferred']} excluded={hum['n_excluded']}")
    print("=" * 72)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="GNN VAAMR reliability battery harness")
    ap.add_argument('arm', nargs='?', default='A0', help="arm name (A0 wired)")
    ap.add_argument('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    run_arm(args.arm, output_dir=args.output_dir)
