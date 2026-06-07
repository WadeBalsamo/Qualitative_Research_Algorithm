"""
tests/unit/test_probe_classifier.py
-----------------------------------
Hermetic tests for classification_tools/probe_classifier.py — the LLM-free, gated,
abstention-aware VAAMR scaler (per-rater ensemble; methodology §8.6).

No network / no model download: synthetic embeddings are passed directly via the
``embeddings=`` hook. The human axis (qra.db consensus) is patched so the gate scores the
in-DataFrame human subset.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools import probe_classifier as pc  # noqa: E402
from classification_tools.probe_classifier import ProbeConfig, ProbeModel  # noqa: E402

RATERS = ['google/gemma-4-31b', 'nvidia/nemotron-3-nano-30b', 'qwen/qwen3-next-80b']


def _synth(n_participants=12, dim=8, separable=True, seed=0):
    """Synthetic master_segments-style frame + {seg_id: embedding}.

    Each participant carries one segment per VAAMR stage 0..4 plus one "No code"
    (final_label null, ballots ABSTAIN). Half the participants are in the human subset.
    When ``separable``, the embedding has a strong signal on the class's own dimension so a
    linear probe recovers the stage near-perfectly (the gate should pass).
    """
    rng = np.random.default_rng(seed)
    rows, emb = [], {}
    for p in range(n_participants):
        pid = f'P{p:02d}'
        sess = f'c1p{p}s1'
        is_human = p < (n_participants // 2)
        for stage in list(range(5)) + [None]:
            sid = f'{pid}_{stage}'
            base = 5 if stage is None else stage
            if stage is None:
                votes = [{'rater': r, 'vote': 'ABSTAIN', 'stage': None} for r in RATERS]
                prim, final, hlab = np.nan, np.nan, (-1 if is_human else np.nan)
            else:
                votes = [{'rater': r, 'vote': 'CODED', 'stage': stage} for r in RATERS]
                prim, final, hlab = stage, stage, (stage if is_human else np.nan)
            rows.append(dict(segment_id=sid, participant_id=pid, session_id=sess,
                             speaker='participant', text=f'{sid} text',
                             primary_stage=prim, final_label=final,
                             rater_votes=json.dumps(votes),
                             in_human_coded_subset=is_human, human_label=hlab))
            v = (rng.standard_normal(dim) * 0.05).astype('float32')
            if separable:
                v[base] += 3.0
            emb[sid] = v
        # a therapist segment (must be excluded from VAAMR)
        tid = f'{pid}_t'
        rows.append(dict(segment_id=tid, participant_id=pid, session_id=sess,
                         speaker='therapist', text='therapist', primary_stage=np.nan,
                         final_label=np.nan, rater_votes=np.nan,
                         in_human_coded_subset=False, human_label=np.nan))
        emb[tid] = rng.standard_normal(dim).astype('float32')
    return pd.DataFrame(rows), emb


def _single_rater(df):
    """Rewrite rater_votes so only ONE distinct rater is present (forces A1n fallback)."""
    df = df.copy()
    def _rw(rv):
        if not isinstance(rv, str):
            return rv
        only = [d for d in json.loads(rv) if d.get('rater') == RATERS[0]]
        return json.dumps(only)
    df['rater_votes'] = df['rater_votes'].map(_rw)
    return df


# patch the qra.db human axis to honour the in-DataFrame human subset (no DB in unit tests)
def _no_db_human(df, output_dir):
    return df


class TestModeSelection(unittest.TestCase):
    def test_ensemble_when_three_raters(self):
        df, _ = _synth()
        use_ens, raters = pc._resolve_mode(df, ProbeConfig())
        self.assertTrue(use_ens)
        self.assertEqual(set(raters), set(RATERS))

    def test_fallback_when_one_rater(self):
        df = _single_rater(_synth()[0])
        use_ens, _ = pc._resolve_mode(df, ProbeConfig())
        self.assertFalse(use_ens)

    def test_default_config_is_the_winner(self):
        c = ProbeConfig()
        self.assertTrue(c.ensemble)
        self.assertEqual(c.n_classes, 6)
        self.assertEqual(c.class_weight, 'balanced')
        self.assertEqual(c.C, 4.0)


class TestTrain(unittest.TestCase):
    def test_ensemble_fits_one_probe_per_rater(self):
        df, emb = _synth()
        with tempfile.TemporaryDirectory() as d:
            model = pc.train_probe(df, d, ProbeConfig(), embeddings=emb)
        self.assertEqual(len(model.rater_probes), 3)
        self.assertEqual(set(model.raters), set(RATERS))
        self.assertEqual(model.n_classes, 6)

    def test_fallback_fits_single_probe(self):
        df, emb = _synth()
        df = _single_rater(df)
        with tempfile.TemporaryDirectory() as d:
            model = pc.train_probe(df, d, ProbeConfig(), embeddings=emb)
        self.assertEqual(len(model.rater_probes), 1)
        self.assertEqual(model.raters, [])

    def test_persists_model_and_manifest(self):
        df, emb = _synth()
        with tempfile.TemporaryDirectory() as d:
            pc.train_probe(df, d, ProbeConfig(), embeddings=emb)
            from process import output_paths as _paths
            mdir = _paths.probe_model_dir(d)
            self.assertTrue(os.path.isfile(os.path.join(mdir, pc.MODEL_FILENAME)))
            with open(os.path.join(mdir, pc.MANIFEST_FILENAME)) as _f:
                man = json.load(_f)
            self.assertEqual(man['n_classes'], 6)
            self.assertTrue(man['ensemble'])
            self.assertEqual(man['training_label_provenance'], 'per_rater_ensemble')
            # round-trip
            reloaded = pc.load_probe_model(d)
            self.assertEqual(len(reloaded.rater_probes), 3)

    def test_calibration_sets_finite_temperature(self):
        df, emb = _synth()
        with tempfile.TemporaryDirectory() as d:
            model = pc.train_probe(df, d, ProbeConfig(calibrate=True), embeddings=emb)
        self.assertTrue(0.0 < model.temperature < 1000.0)


class TestPredict(unittest.TestCase):
    def _model(self):
        df, emb = _synth()
        with tempfile.TemporaryDirectory() as d:
            return pc.train_probe(df, d, ProbeConfig(), embeddings=emb), emb

    def test_shapes_and_mixture(self):
        model, emb = self._model()
        X = np.stack([emb['P00_0'], emb['P00_2'], emb['P00_4']])
        pred, conf, abstain, mix = model.predict(X)
        self.assertEqual(pred.shape, (3,))
        self.assertEqual(mix.shape, (3, 5))
        self.assertTrue(np.allclose(mix.sum(axis=1), 1.0, atol=1e-6))
        self.assertTrue(((conf > 0) & (conf <= 1.0 + 1e-9)).all())

    def test_no_code_predicts_class5_and_abstains(self):
        model, emb = self._model()
        pred, conf, abstain, _ = model.predict(np.stack([emb['P00_None']]))
        self.assertEqual(int(pred[0]), 5)
        self.assertTrue(bool(abstain[0]))

    def test_separable_recovers_stage(self):
        model, emb = self._model()
        for stage in range(5):
            pred, _, _, _ = model.predict(np.stack([emb[f'P00_{stage}']]))
            self.assertEqual(int(pred[0]), stage)

    def test_abstain_below_global_floor(self):
        # deterministic: a stub probe with a moderate (0.5) top class, floor above it → defer
        class _StubClf:
            classes_ = np.arange(6)
            def predict_proba(self, X):
                p = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
                return np.tile(p, (len(X), 1))
        model = ProbeModel(rater_probes=[_StubClf()], raters=[], n_classes=6,
                           temperature=1.0, abstain_threshold=0.8, abstain_per_stage=None)
        pred, conf, abstain, _ = model.predict(np.zeros((1, 8), dtype='float32'))
        self.assertEqual(int(pred[0]), 0)
        self.assertLess(conf[0], 0.8)
        self.assertTrue(bool(abstain[0]))
        # and with a low floor it does NOT abstain
        model.abstain_threshold = 0.4
        _, _, abstain2, _ = model.predict(np.zeros((1, 8), dtype='float32'))
        self.assertFalse(bool(abstain2[0]))


class TestGate(unittest.TestCase):
    def test_ready_when_separable(self):
        df, emb = _synth()
        with mock.patch('analysis.irr_join.populate_human_columns', side_effect=_no_db_human):
            with tempfile.TemporaryDirectory() as d:
                v = pc.evaluate_probe(df, d, ProbeConfig(), embeddings=emb)
                self.assertTrue(v['ready_for_scaling'])
                self.assertGreaterEqual(v['probe_human_kappa'], 0.33)
                self.assertTrue(v['rare_ok'])
                # gate + report written
                from process import output_paths as _paths
                self.assertTrue(os.path.isfile(
                    os.path.join(_paths.probe_data_dir(d), pc.GATE_FILENAME)))
                self.assertTrue(os.path.isfile(
                    os.path.join(_paths.reports_classifier_dir(d), 'probe_validation.txt')))
                self.assertTrue(pc.probe_gate_ready(d))

    def test_not_ready_when_random(self):
        df, emb = _synth(separable=False, seed=3)
        with mock.patch('analysis.irr_join.populate_human_columns', side_effect=_no_db_human):
            with tempfile.TemporaryDirectory() as d:
                v = pc.evaluate_probe(df, d, ProbeConfig(), embeddings=emb)
                self.assertFalse(v['ready_for_scaling'])
                self.assertFalse(pc.probe_gate_ready(d))


class TestClassifyTargets(unittest.TestCase):
    def test_only_unlabeled_participants_targeted(self):
        # one extra participant whose segments are UNLABELED (no final_label/primary_stage)
        df, emb = _synth(n_participants=8)
        extra = []
        for i, stage in enumerate(range(3)):
            sid = f'NEW_{i}'
            extra.append(dict(segment_id=sid, participant_id='PNEW', session_id='c1pNs1',
                              speaker='participant', text='new', primary_stage=np.nan,
                              final_label=np.nan, rater_votes=np.nan,
                              in_human_coded_subset=False, human_label=np.nan))
            v = (np.zeros(8, dtype='float32')); v[stage] += 3.0
            emb[sid] = v
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        with tempfile.TemporaryDirectory() as d:
            pc.train_probe(df, d, ProbeConfig(), embeddings=emb)
            written = {}
            # capture overlay rather than touching a real DB
            import process.classifications_io as cio
            with mock.patch.object(cio, 'merge_probe_overlay', create=True,
                                   side_effect=lambda od, segs: written.update({s.segment_id: s for s in segs})):
                n = pc.classify_with_probe(df, d, ProbeConfig(), embeddings=emb)
        # only the 3 brand-new unlabeled participant segments are filled
        self.assertEqual(n, 3)
        self.assertEqual(set(written), {'NEW_0', 'NEW_1', 'NEW_2'})
        self.assertEqual(int(written['NEW_0'].probe_pred), 0)


if __name__ == '__main__':
    unittest.main()
