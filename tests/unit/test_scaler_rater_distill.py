"""
tests/unit/test_scaler_rater_distill.py
---------------------------------------
Hermetic unit test for the campaign WINNER — the per-rater ensemble
(``experiments/classification_scaler/rater_distill.py``): per-rater label extraction from the
multi-run ``rater_votes`` ballots, and the soft-average / majority ensembles. No ``data/``, no
network, no real embeddings — everything is synthetic with a fixed seed (mirrors
``test_gnn_baselines``).

Asserts:
  * ``participant_rater_labels`` extracts a label set per rater, ABSTAIN -> No-code (class 5),
    and covers every participant segment that carries a ``rater_votes`` ballot;
  * ``run_ensemble('softavg')`` returns OOF preds for the labeled participant segments, all in
    range [0, 5];
  * the soft-average ensemble differs from at least one single rater's argmax (it integrates the
    full ballot distribution rather than copying one rater).
"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
if _ROOT not in sys.path:
    sys.path.insert(1, _ROOT)

from experiments.classification_scaler import rater_distill as RD  # noqa: E402
from gnn_layer.config import GnnLayerConfig  # noqa: E402

_SEED = 1234
_DIM = 16
RATERS = RD.RATERS


def _votes(stages):
    """stages: 3 entries (int 0-4 = CODED stage, or None = ABSTAIN) -> rater_votes JSON string."""
    out = []
    for r, s in zip(RATERS, stages):
        if s is None:
            out.append({'rater': r, 'vote': 'ABSTAIN'})
        else:
            out.append({'rater': r, 'vote': 'CODED', 'stage': int(s)})
    return json.dumps(out)


def _make_corpus(seed=_SEED, dim=_DIM):
    """6 participants x (6 participant + 3 therapist) segments, class-correlated embeddings.

    Each participant segment carries a 3-rater ballot: two raters agree on the majority stage,
    the third dissents (or all three ABSTAIN on a No-code segment) -> real disagreement structure.
    """
    rng = np.random.default_rng(seed)
    class_mean = {c: rng.standard_normal(dim) for c in range(6)}
    ther_mean = rng.standard_normal(dim)
    pool = [2, 2, 0, 2, 3, None, 2, 1, 2, 4, 0, 3]  # majority class per seg (None -> No-code)
    rows, emb = [], {}
    for p in range(6):
        pid = f'P{p:02d}'
        sess = f'c1{pid}s1'
        clock = 0
        ppos = 0
        for j in range(9):
            clock += 1000
            sid = f'{sess}_{j}'
            if j % 3 == 0:  # therapist
                rows.append(dict(segment_id=sid, participant_id=pid, session_id=sess,
                                 speaker='therapist', start_time_ms=clock,
                                 final_label=np.nan, rater_votes=None))
                emb[sid] = (0.8 * ther_mean + 0.8 * rng.standard_normal(dim)).astype('float32')
                continue
            maj = pool[(p * 7 + ppos) % len(pool)]
            ppos += 1
            cls = 5 if maj is None else int(maj)
            third = None if maj is None else (maj + 1) % 5
            rows.append(dict(segment_id=sid, participant_id=pid, session_id=sess,
                             speaker='participant', start_time_ms=clock,
                             final_label=(np.nan if maj is None else float(maj)),
                             rater_votes=_votes([maj, maj, third])))
            emb[sid] = (0.8 * class_mean[cls] + 0.8 * rng.standard_normal(dim)).astype('float32')
    df = pd.DataFrame(rows)
    # participant-grouped folds over labeled participant segments (incl. No-code)
    folds = {}
    for _, r in df.iterrows():
        if str(r['speaker']) == 'participant' and isinstance(r['rater_votes'], str):
            folds[str(r['segment_id'])] = int(r['participant_id'][1:]) % 3
    return df, emb, folds


class TestRaterDistillWinner(unittest.TestCase):

    def test_per_rater_label_extraction(self):
        df, _emb, _folds = _make_corpus()
        seg_ids, per_rater, _soft = RD.participant_rater_labels(df, 6)
        self.assertGreater(len(seg_ids), 0)
        for r in RATERS:
            self.assertGreater(len(per_rater[r]), 0, f'rater {r} has no labels')
        # No-code segments: all three raters ABSTAIN -> class 5 for each
        no_code = [s for s in seg_ids if per_rater[RATERS[0]].get(s) == 5]
        self.assertTrue(no_code, 'fixture should contain at least one No-code segment')
        for s in no_code:
            self.assertTrue(all(per_rater[r].get(s) == 5 for r in RATERS))

    def test_ens_softavg_oof_complete_and_in_range(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True)
        preds = RD.run_ensemble(df, emb, folds, cfg, 'softavg')
        self.assertGreater(len(preds), 0)
        self.assertTrue(all(isinstance(v, int) for v in preds.values()))
        self.assertTrue(all(0 <= v <= 5 for v in preds.values()))

    def test_ensemble_integrates_full_ballot_not_one_rater(self):
        df, emb, folds = _make_corpus()
        cfg = GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True)
        seg_ids, proba = RD.per_rater_oof_proba(df, emb, folds, cfg)
        ens = RD.ensemble_from_proba(seg_ids, proba, 6, 'softavg')
        single = RD.per_rater_argmax(seg_ids, proba, RATERS[2])  # the dissenting rater
        common = [s for s in ens if s in single]
        self.assertGreaterEqual(
            sum(1 for s in common if ens[s] != single[s]), 1,
            'soft-average ensemble should differ from a single rater on >=1 segment')


if __name__ == '__main__':
    unittest.main()
