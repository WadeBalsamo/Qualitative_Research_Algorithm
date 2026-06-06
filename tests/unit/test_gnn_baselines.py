"""
tests/unit/test_gnn_baselines.py
--------------------------------
Hermetic unit tests for the non-GNN VAAMR reliability baselines
(``experiments/gnn_reliability/baselines.py``): the linear probe (arms A1/A1w/A1n)
and Correct-&-Smooth (arm A2). No ``data/``, no network, no real embeddings —
everything is synthetic with a fixed seed.

Asserts:
  * probe returns OOF preds for ALL labeled participant segments, every pred in range;
  * ``vaamr_class_balance=True`` changes predictions vs False;
  * ``vaamr_n_classes=6`` emits some class-5 ("No code") predictions + folds the
    No-code rows (absent from the harness ``folds``) by participant;
  * Correct-&-Smooth runs end-to-end, returns valid preds, and DIFFERS from the bare
    probe on at least one node (propagation had an effect).
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)
# baselines.py lives in experiments/gnn_reliability/ with no __init__.py (harness owns it),
# so import it as a top-level module by adding its directory to sys.path.
sys.path.insert(0, os.path.join(_QRA_ROOT, 'experiments', 'gnn_reliability'))

import baselines  # noqa: E402
from gnn_layer.config import GnnLayerConfig  # noqa: E402

_SEED = 1234
_DIM = 16

# Skewed pool (lots of class 2; a few None=No-code) -> class imbalance so balancing bites.
_LABEL_POOL = [2, 2, 0, 2, 3, None, 2, 1, 2, 4, 2, None, 0, 2, 3, None]


def _make_corpus(seed=_SEED, dim=_DIM):
    """6 participants x (7 participant + 2 therapist) segments.

    Participant 5 is ENTIRELY No-code (exercises the orphan-participant fold path
    under 6-class). Embeddings are class-correlated (a per-class mean + noise) so the
    probe is non-degenerate and C&S has real structure to smooth over.

    Returns (df_all, embeddings, folds) where ``folds`` is participant-grouped over
    the LABELED participant segments only (mirrors the harness).
    """
    rng = np.random.default_rng(seed)
    class_mean = {c: rng.standard_normal(dim) for c in range(6)}  # class 5 = No-code
    ther_mean = rng.standard_normal(dim)

    rows = []
    embeddings = {}
    n_part = 6
    for p in range(n_part):
        pid = f'P{p:02d}'
        sess = f'c1{pid}s1'
        clock = 0
        ppos = 0
        for j in range(9):  # interleave: positions 0,3,6 are therapist; rest participant
            clock += 1000
            sid = f'{sess}_{j}'
            is_ther = (j % 3 == 0)
            if is_ther:
                rows.append(dict(segment_id=sid, participant_id=pid, session_id=sess,
                                 speaker='therapist', start_time_ms=clock,
                                 final_label=np.nan))
                embeddings[sid] = (0.8 * ther_mean + 0.8 * rng.standard_normal(dim)).astype('float32')
            else:
                if p == n_part - 1:
                    lab = None  # participant 5 = entirely No-code
                else:
                    lab = _LABEL_POOL[(p * 7 + ppos) % len(_LABEL_POOL)]
                ppos += 1
                cls = 5 if lab is None else int(lab)
                rows.append(dict(segment_id=sid, participant_id=pid, session_id=sess,
                                 speaker='participant', start_time_ms=clock,
                                 final_label=(np.nan if lab is None else float(lab))))
                embeddings[sid] = (0.8 * class_mean[cls] + 0.8 * rng.standard_normal(dim)).astype('float32')
    df = pd.DataFrame(rows)

    # participant-grouped folds over the LABELED participant segments only (3 folds)
    folds = {}
    for _, r in df.iterrows():
        if str(r['speaker']) == 'participant' and not pd.isna(r['final_label']):
            folds[str(r['segment_id'])] = int(r['participant_id'][1:]) % 3
    return df, embeddings, folds


def _labeled_part_ids(df, n_classes):
    """Expected target seg_ids for a given n_classes (mirrors baselines._prepare_labeled)."""
    out = []
    for _, r in df.iterrows():
        if str(r['speaker']) != 'participant':
            continue
        if not pd.isna(r['final_label']):
            out.append(str(r['segment_id']))
        elif n_classes >= 6:
            out.append(str(r['segment_id']))
    return set(out)


class TestLinearProbe(unittest.TestCase):

    def test_oof_complete_and_in_range(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(knn_k=8, cache_embeddings=False,
                             vaamr_n_classes=5, vaamr_class_balance=False)
        preds = baselines.run_linear_probe(df, emb, folds, cfg)
        self.assertEqual(set(preds), _labeled_part_ids(df, 5))
        self.assertTrue(all(isinstance(v, int) for v in preds.values()))
        self.assertTrue(all(0 <= v <= 4 for v in preds.values()))

    def test_class_balance_changes_predictions(self):
        df, emb, folds = _make_corpus()
        base = GnnLayerConfig(knn_k=8, cache_embeddings=False, vaamr_n_classes=5)
        unbal = baselines.run_linear_probe(
            df, emb, folds, GnnLayerConfig(**{**base.__dict__, 'vaamr_class_balance': False}))
        bal = baselines.run_linear_probe(
            df, emb, folds, GnnLayerConfig(**{**base.__dict__, 'vaamr_class_balance': True}))
        self.assertEqual(set(unbal), set(bal))
        self.assertNotEqual(unbal, bal, "balanced class weights should change >=1 prediction")

    def test_six_class_emits_no_code_and_folds_orphans(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(knn_k=8, cache_embeddings=False,
                             vaamr_n_classes=6, vaamr_class_balance=True)
        preds = baselines.run_linear_probe(df, emb, folds, cfg)
        # No-code participant rows (incl. entirely-No-code participant 5) are folded + predicted
        self.assertEqual(set(preds), _labeled_part_ids(df, 6))
        self.assertTrue(all(0 <= v <= 5 for v in preds.values()))
        self.assertGreaterEqual(sum(1 for v in preds.values() if v == 5), 1,
                                "6-class probe should predict the No-code class at least once")


class TestCorrectAndSmooth(unittest.TestCase):

    def test_runs_valid_and_differs_from_probe(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(knn_k=8, cache_embeddings=False,
                             vaamr_n_classes=5, vaamr_class_balance=False)
        probe = baselines.run_linear_probe(df, emb, folds, cfg)
        cs = baselines.run_correct_smooth(df, emb, folds, cfg)
        self.assertEqual(set(cs), set(probe))
        self.assertTrue(all(0 <= v <= 4 for v in cs.values()))
        diffs = [s for s in cs if cs[s] != probe[s]]
        self.assertGreaterEqual(len(diffs), 1,
                                "C&S propagation should change >=1 node vs the bare probe")

    def test_six_class_correct_smooth_valid(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(knn_k=8, cache_embeddings=False,
                             vaamr_n_classes=6, vaamr_class_balance=True)
        cs = baselines.run_correct_smooth(df, emb, folds, cfg)
        self.assertEqual(set(cs), _labeled_part_ids(df, 6))
        self.assertTrue(all(0 <= v <= 5 for v in cs.values()))


if __name__ == '__main__':
    unittest.main()
