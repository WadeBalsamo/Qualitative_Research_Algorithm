"""
process/assembly/mindfulbert_dataset.py
---------------------------------------
Track C — the MindfulBERT training-set builder (the end-goal artifact).

QRA's GNN work exists to produce a **training dataset to fine-tune MindfulBERT** — a model
that does not merely classify VAAMR but *predicts which therapist language progresses a
participant across VAAMR stages*. This module assembles that dataset.

The unit is the **cue block** (FROM participant → therapist cue → TO participant). Each
example pairs the preceding participant context + VAAMR state and the therapist cue language
with a label:

  PRIMARY label  = the OBSERVED Δprogression of the block (signed E[stage] change + a
                   categorical direction). This is the label of record (master plan D6/D8).
  AUGMENTATION   = an OPTIONAL, provenance-tagged GNN model-counterfactual "would-progress"
                   channel (C3), produced ONLY from a gate-passing model and never silently
                   merged with the observed labels.

Every example carries per-example **provenance + confidence** (label source tier, abstention
flag, gate verdict). The augmentation channel is kept only if a held-out ablation (C4) shows
it improves a lightweight progression-prediction proxy by more than ``augmentation_min_gain``
— otherwise it is dropped. A datasheet (C5) records the provenance mix, gate status, the
augmentation ablation result, and the n≈32 / observational / non-causal caveats.

Output: ``02_meta/training_data/mindfulbert_dataset.jsonl`` + ``mindfulbert_datasheet.{json,txt}``.

GPU (D11): the optional C4 proxy is sklearn/CPU by design; only the upstream counterfactual
channel (gnn_layer/influence.py) is GPU-relevant and follows the device mandate there.
"""

import datetime
import json
import os
from collections import Counter
from typing import Dict, List, Optional

from .. import output_paths as _paths

DATASET_VERSION = '1.0'
PURER_NAMES = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
               3: 'Education', 4: 'Reinforcement'}
# Provenance precedence (higher = stronger). An example's tier is the WEAKEST of its
# two VAAMR endpoints — the dataset never claims more certainty than its weakest label.
_TIER_RANK = {'adjudicated': 3, 'human_consensus': 2, 'gnn_consensus': 1, 'llm_zero_shot': 0}
_RANK_TIER = {v: k for k, v in _TIER_RANK.items()}
_PROGRESS_DEADBAND = 0.15  # |Δprogression| below this is "stayed" (matches analysis/mechanism)


def _coerce_int(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == f else None


def _seg_index(df_all):
    """segment_id → row dict with the fields the builder needs."""
    out: Dict[str, dict] = {}
    has_coord = 'progression_coord' in df_all.columns
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id', ''))
        if not sid:
            continue
        coord = None
        if has_coord:
            try:
                c = float(r.get('progression_coord'))
                coord = c if c == c else None
            except (TypeError, ValueError):
                coord = None
        out[sid] = {
            'text': str(r.get('text', '') or ''),
            'speaker': str(r.get('speaker', '') or ''),
            'progression_coord': coord,
            'final_label': _coerce_int(r.get('final_label')),
            'final_label_source': r.get('final_label_source'),
            'confidence': r.get('llm_confidence_primary'),
            'purer': _coerce_int(r.get('purer_primary')),
            'participant_id': r.get('participant_id'),
            'session_number': r.get('session_number'),
            'gnn_vaamr_abstain': bool(r.get('gnn_vaamr_abstain', False)),
        }
    return out


def _tier_of(source) -> Optional[str]:
    if source is None or (isinstance(source, float) and source != source):
        return None
    s = str(source)
    return s if s in _TIER_RANK else 'llm_zero_shot'


def _direction(delta: float) -> str:
    if delta > _PROGRESS_DEADBAND:
        return 'advanced'
    if delta < -_PROGRESS_DEADBAND:
        return 'regressed'
    return 'stayed'


def _build_examples(df_all, gate_passed: bool) -> List[dict]:
    """C1/C2 — one example per mediated cue block with observed-Δprogression labels + provenance."""
    from gnn_layer.inference import build_cue_blocks_with_segments

    lookup = _seg_index(df_all)
    blocks = build_cue_blocks_with_segments(df_all)
    examples = []
    for b in blocks:
        ther = b.get('therapist_seg_ids') or []
        if not ther:
            continue
        fl = lookup.get(b['from_seg_id'])
        tl = lookup.get(b['to_seg_id'])
        if not fl or not tl:
            continue

        # Signed Δprogression: prefer the continuous coordinate, fall back to stage difference.
        if fl['progression_coord'] is not None and tl['progression_coord'] is not None:
            delta = tl['progression_coord'] - fl['progression_coord']
            label_basis = 'progression_coord'
        else:
            delta = float(b['to_stage'] - b['from_stage'])
            label_basis = 'stage_difference'

        cue_text = ' '.join(lookup[s]['text'] for s in ther if s in lookup).strip()
        purers = [lookup[s]['purer'] for s in ther if s in lookup and lookup[s]['purer'] is not None]
        dom = Counter(purers).most_common(1)[0][0] if purers else None

        from_tier = _tier_of(fl['final_label_source'])
        to_tier = _tier_of(tl['final_label_source'])
        ranks = [_TIER_RANK.get(t, 0) for t in (from_tier, to_tier) if t is not None]
        prov_tier = _RANK_TIER.get(min(ranks), 'llm_zero_shot') if ranks else 'llm_zero_shot'

        examples.append({
            'cue_block_id': f"{b['from_seg_id']}__{b['to_seg_id']}",
            'session_id': b['session_id'],
            'participant_id': fl['participant_id'],
            'session_number': fl['session_number'],
            'context_text': fl['text'],
            'cue_text': cue_text,
            'from_stage': int(b['from_stage']),
            'to_stage': int(b['to_stage']),
            'dominant_purer': (int(dom) if dom is not None else None),
            'dominant_purer_name': (PURER_NAMES.get(int(dom)) if dom is not None else None),
            'n_therapist_segments': len(ther),
            'n_cue_words': len(cue_text.split()),
            # ---- PRIMARY label (observed Δprogression — label of record) ----
            'delta_progression': round(float(delta), 5),
            'direction': _direction(float(delta)),
            'label_basis': label_basis,
            # ---- per-example provenance + confidence (C2) ----
            'provenance': {
                'tier': prov_tier,
                'from_label_source': from_tier,
                'to_label_source': to_tier,
                'gnn_abstain': bool(fl['gnn_vaamr_abstain'] or tl['gnn_vaamr_abstain']),
                'gate_passed': bool(gate_passed),
            },
        })
    return examples


def _attach_augmentation(examples, output_dir: str) -> int:
    """C3 — attach the GNN-counterfactual 'would-progress' channel (separate, provenance-tagged).

    Reads the per-move counterfactual influence (gnn_layer/influence.py output) and tags each
    example with the model's would-progress estimate for its dominant move. Returns the number
    of examples that received the channel. Never overwrites the observed label.
    """
    import pandas as pd
    path = os.path.join(_paths.gnn_data_dir(output_dir), 'gnn_counterfactual_influence.csv')
    if not os.path.isfile(path):
        return 0
    try:
        idf = pd.read_csv(path)
    except Exception:
        return 0
    if 'move' not in idf.columns or 'mean_influence' not in idf.columns:
        return 0
    infl_by_move = {int(r['move']): float(r['mean_influence']) for _, r in idf.iterrows()}
    n = 0
    for ex in examples:
        m = ex.get('dominant_purer')
        if m is not None and m in infl_by_move:
            ex['augmentation'] = {
                'provenance': 'gnn_counterfactual',
                'would_progress': round(infl_by_move[m], 5),
            }
            n += 1
    return n


def _augmentation_ablation(examples, config) -> dict:
    """C4 — does the augmentation channel improve a held-out progression-prediction proxy?

    Trains a lightweight logistic-regression proxy to predict the observed binary outcome
    (advanced vs not) from cue features, WITH vs WITHOUT the counterfactual channel, using
    participant-grouped cross-validation (no participant leaks across folds). Returns the
    base/augmented held-out accuracy, the gain, and a retain verdict. Augmentation is RETAINED
    only if gain > ``augmentation_min_gain``.
    """
    import numpy as np
    rows = [ex for ex in examples if ex.get('augmentation') is not None
            and ex.get('participant_id') is not None]
    out = {'status': None, 'base_accuracy': None, 'augmented_accuracy': None,
           'gain': None, 'min_gain': float(getattr(config, 'augmentation_min_gain', 0.0)),
           'retain': False, 'n_examples': len(rows)}
    if len(rows) < 12:
        out['status'] = 'inconclusive: too few augmented examples (<12)'
        return out
    y = np.array([1 if ex['direction'] == 'advanced' else 0 for ex in rows])
    if y.sum() == 0 or y.sum() == len(y):
        out['status'] = 'inconclusive: single-class outcome'
        return out

    def _onehot(vals, k):
        M = np.zeros((len(vals), k), dtype=float)
        for i, v in enumerate(vals):
            if v is not None and 0 <= int(v) < k:
                M[i, int(v)] = 1.0
        return M

    from_oh = _onehot([ex['from_stage'] for ex in rows], 5)
    purer_oh = _onehot([ex['dominant_purer'] for ex in rows], 5)
    nwords = np.array([[ex['n_cue_words'], ex['n_therapist_segments']] for ex in rows], dtype=float)
    aug = np.array([[ex['augmentation']['would_progress']] for ex in rows], dtype=float)
    X_base = np.hstack([from_oh, purer_oh, nwords])
    X_aug = np.hstack([X_base, aug])
    groups = np.array([str(ex['participant_id']) for ex in rows])

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        n_groups = len(set(groups))
        if n_groups < 2:
            out['status'] = 'inconclusive: <2 participant groups'
            return out
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)

        def _cv_acc(X):
            correct = total = 0
            for tr, te in gkf.split(X, y, groups):
                if len(set(y[tr])) < 2:
                    continue
                clf = LogisticRegression(max_iter=500)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                correct += int((pred == y[te]).sum())
                total += len(te)
            return (correct / total) if total else None

        base = _cv_acc(X_base)
        augd = _cv_acc(X_aug)
    except Exception as e:
        out['status'] = f'inconclusive: proxy failed ({e})'
        return out
    if base is None or augd is None:
        out['status'] = 'inconclusive: no valid CV folds'
        return out
    gain = augd - base
    out.update({'status': 'ok', 'base_accuracy': round(base, 4),
                'augmented_accuracy': round(augd, 4), 'gain': round(gain, 4),
                'retain': bool(gain > float(out['min_gain']))})
    return out


def _write_datasheet(examples, datasheet: dict, output_dir: str) -> str:
    """C5 — human-readable datasheet → 02_meta/training_data/mindfulbert_datasheet.txt."""
    W = 78
    L = ["=" * W, "MINDFULBERT TRAINING-SET DATASHEET", "=" * W, ""]
    L.append(f"  dataset version : {datasheet['dataset_version']}")
    L.append(f"  created         : {datasheet['created']}")
    L.append(f"  examples        : {datasheet['n_examples']} cue blocks")
    L.append(f"  participants    : {datasheet['n_participants']}   sessions: {datasheet['n_sessions']}")
    L.append(f"  label basis     : {datasheet['label_basis']}")
    L.append("")
    L.append("  TASK: predict which therapist cue language progresses a participant across VAAMR")
    L.append("  stages. Unit = cue block (FROM participant → therapist cue → TO participant).")
    L.append("  PRIMARY label = OBSERVED Δprogression (signed E[stage] change + direction). This")
    L.append("  is the label of record; the GNN never overrides it.")
    L.append("")
    L.append("-" * W)
    L.append("  OUTCOME (direction) DISTRIBUTION")
    L.append("-" * W)
    for k, v in datasheet['direction_distribution'].items():
        L.append(f"    {k:<12} {v}")
    L.append("")
    L.append("-" * W)
    L.append("  PROVENANCE MIX (weakest endpoint per example)")
    L.append("-" * W)
    for k, v in datasheet['provenance_mix'].items():
        L.append(f"    {k:<18} {v}")
    L.append(f"    examples with a GNN-abstained endpoint: {datasheet['n_abstained']}")
    L.append("")
    L.append("-" * W)
    L.append("  GATE STATUS & AUGMENTATION (C3/C4)")
    L.append("-" * W)
    L.append(f"    reliability gate passed : {datasheet['gate_passed']}")
    aug = datasheet['augmentation']
    if not aug.get('enabled'):
        L.append("    augmentation            : DISABLED (observed labels only)")
    elif not datasheet['gate_passed']:
        L.append("    augmentation            : SUPPRESSED (gate not passed — never augment off an")
        L.append("                              un-gated model)")
    else:
        L.append(f"    augmentation channel    : attached to {aug.get('n_augmented', 0)} examples "
                 f"(provenance 'gnn_counterfactual')")
        abl = aug.get('ablation', {})
        L.append(f"    C4 ablation             : {abl.get('status')}")
        if abl.get('status') == 'ok':
            L.append(f"      base acc={abl['base_accuracy']}  augmented acc={abl['augmented_accuracy']}  "
                     f"gain={abl['gain']:+.4f}  (min {abl['min_gain']:+.4f})")
        L.append(f"    RETAINED in dataset     : {'YES' if aug.get('retained') else 'NO'}")
        if not aug.get('retained'):
            L.append("      (augmentation did not clear its retention threshold → dropped from the")
            L.append("       exported examples; observed labels stand alone.)")
    L.append("")
    L.append("-" * W)
    L.append("  CAVEATS")
    L.append("-" * W)
    L.append("    • n≈32, single-arm, unblinded, observational. Labels are ASSOCIATIONAL, not")
    L.append("      causal — the elicitation confound (PURER inquiry elicits the language VAAMR")
    L.append("      scores) is not removed by this dataset (methodology §9.2/§9.4).")
    L.append("    • The augmentation channel is MODEL-COUNTERFACTUAL SENSITIVITY, not ground truth;")
    L.append("      it is retained only under the C4 held-out ablation and stays provenance-tagged.")
    L.append("    • Train/eval splits should respect the participant grouping to avoid leakage.")
    L.append("")
    tdir = _paths.training_data_dir(output_dir)
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, 'mindfulbert_datasheet.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def build_mindfulbert_dataset(df_all, output_dir: str, config=None,
                              gate_passed: Optional[bool] = None, verbose: bool = True) -> dict:
    """Assemble + export the versioned MindfulBERT (cue language → Δprogression) dataset.

    ``gate_passed`` overrides the persisted gate verdict (None → read it). Augmentation (C3)
    is attached only when ``config.augmentation_enabled`` AND the gate passed, and RETAINED in
    the exported examples only when the C4 ablation clears ``augmentation_min_gain``.

    Returns {files_written, n_examples, status}.
    """
    def _log(msg):
        if verbose:
            print(f"  [mindfulbert] {msg}")

    files: List[str] = []
    if df_all is None or len(df_all) == 0 or 'segment_id' not in df_all.columns:
        return {'files_written': files, 'n_examples': 0, 'status': 'skipped: empty/invalid dataframe'}

    if gate_passed is None:
        try:
            from gnn_layer.validation import gate_ready_for_scaling
            gate_passed = gate_ready_for_scaling(output_dir)
        except Exception:
            gate_passed = False
    gate_passed = bool(gate_passed)

    examples = _build_examples(df_all, gate_passed)
    if not examples:
        return {'files_written': files, 'n_examples': 0, 'status': 'skipped: no mediated cue blocks'}

    # ---- C3/C4: augmentation channel (gate-passing only) ----
    aug_meta = {'enabled': bool(getattr(config, 'augmentation_enabled', False)),
                'n_augmented': 0, 'retained': False, 'ablation': {}}
    if aug_meta['enabled'] and gate_passed:
        n_aug = _attach_augmentation(examples, output_dir)
        aug_meta['n_augmented'] = n_aug
        if n_aug:
            abl = _augmentation_ablation(examples, config)
            aug_meta['ablation'] = abl
            aug_meta['retained'] = bool(abl.get('retain'))
            _log(f"augmentation ablation: {abl.get('status')} (retain={aug_meta['retained']})")
        if not aug_meta['retained']:
            # Drop the channel from the exported examples (never ship un-validated augmentation).
            for ex in examples:
                ex.pop('augmentation', None)
    elif aug_meta['enabled'] and not gate_passed:
        _log("augmentation requested but the reliability gate has not passed — suppressed.")

    # ---- C5: export dataset JSONL ----
    tdir = _paths.training_data_dir(output_dir)
    os.makedirs(tdir, exist_ok=True)
    jsonl_path = os.path.join(tdir, 'mindfulbert_dataset.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False, default=str) + '\n')
    files.append(jsonl_path)

    # ---- datasheet ----
    prov_mix = Counter(ex['provenance']['tier'] for ex in examples)
    dir_dist = Counter(ex['direction'] for ex in examples)
    n_abstained = sum(1 for ex in examples if ex['provenance']['gnn_abstain'])
    datasheet = {
        'dataset_version': DATASET_VERSION,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_examples': len(examples),
        'n_participants': len({ex['participant_id'] for ex in examples if ex['participant_id'] is not None}),
        'n_sessions': len({ex['session_id'] for ex in examples}),
        'label_basis': Counter(ex['label_basis'] for ex in examples).most_common(1)[0][0],
        'direction_distribution': dict(dir_dist),
        'provenance_mix': dict(prov_mix),
        'n_abstained': n_abstained,
        'gate_passed': gate_passed,
        'augmentation': aug_meta,
    }
    json_path = os.path.join(tdir, 'mindfulbert_datasheet.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(datasheet, f, indent=2, default=str)
    files.append(json_path)
    files.append(_write_datasheet(examples, datasheet, output_dir))

    _log(f"wrote {len(examples)} examples ({dict(dir_dist)}); gate_passed={gate_passed}")
    return {'files_written': files, 'n_examples': len(examples), 'status': 'ok',
            'gate_passed': gate_passed, 'augmentation_retained': aug_meta['retained']}
