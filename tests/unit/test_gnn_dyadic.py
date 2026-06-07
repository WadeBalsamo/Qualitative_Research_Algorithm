"""
tests/unit/test_gnn_dyadic.py
-----------------------------
Unit tests for WS2 — deepened Track-D discovery (gnn_layer/communities.py):
community ↔ VAAMR-stage profile, atypical-exemplar retrieval, and the dyadic routines
(therapist-community → following participant-community) with stability selection.

Hermetic: make_master_df + hand-assigned community labels; no model download, no qra.db.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from gnn_layer import communities as COMM
from gnn_layer.config import GnnLayerConfig


def _cfg(**kw):
    base = dict(community_min_size=2, community_stability_boots=6, seed=1)
    base.update(kw)
    return GnnLayerConfig(**base)


class TestCommunityStageProfile(unittest.TestCase):

    def test_participant_community_has_stage_distribution(self):
        df = make_master_df(n_sessions=3, n_participants=4)
        meta = COMM._seg_meta(df)
        part = [s for s in meta if meta[s]['speaker'] == 'participant']
        ther = [s for s in meta if meta[s]['speaker'] == 'therapist']
        communities = [set(part), set(ther)]
        prof = COMM.community_stage_profile(communities, meta, _cfg())
        self.assertIn(0, prof)                      # the participant community
        self.assertGreater(prof[0]['n_participant_labeled'], 0)
        self.assertTrue(prof[0]['stage_distribution'])
        self.assertIsNotNone(prof[0]['dominant_stage'])


class TestAtypicalExemplars(unittest.TestCase):

    def test_returns_prototypical_and_atypical(self):
        df = make_master_df(n_sessions=3, n_participants=4)
        meta = COMM._seg_meta(df)
        rng = np.random.default_rng(0)
        emb = {s: rng.standard_normal(16).astype('float32') for s in meta}
        part = [s for s in meta if meta[s]['speaker'] == 'participant']
        out = COMM.atypical_exemplars([set(part)], emb, meta, _cfg())
        self.assertIn(0, out)
        self.assertIn('atypical', out[0])
        self.assertIn('prototypical', out[0])


class TestDyadicRoutines(unittest.TestCase):

    def _labels(self, meta):
        # therapists -> community 0, participants -> community 1 (one clean routine 0->1)
        return {s: (0 if meta[s]['speaker'] == 'therapist' else 1) for s in meta}

    def test_routine_formed_and_stability_flagged(self):
        df = make_master_df(n_sessions=3, n_participants=4)
        meta = COMM._seg_meta(df)
        labels = self._labels(meta)
        routines = COMM.dyadic_routines(df, labels, meta, _cfg(), stable_ids={0, 1}, min_count=1)
        self.assertTrue(routines)
        r0 = routines[0]
        self.assertEqual((r0['cue_community'], r0['to_community']), (0, 1))
        self.assertGreaterEqual(r0['count'], 1)
        self.assertTrue(r0['stable'])               # both communities in stable_ids

    def test_unstable_when_community_not_selected(self):
        df = make_master_df(n_sessions=3, n_participants=4)
        meta = COMM._seg_meta(df)
        labels = self._labels(meta)
        routines = COMM.dyadic_routines(df, labels, meta, _cfg(), stable_ids={0}, min_count=1)
        # to_community (1) is not in stable_ids -> routine flagged unstable
        self.assertTrue(all(not r['stable'] for r in routines))


class TestDyadicWriters(unittest.TestCase):

    def test_csv_and_report(self):
        routines = [{'cue_community': 0, 'to_community': 1, 'count': 5, 'selection_freq': 0.9,
                     'mean_delta_prog': 0.12, 'ci_lo': -0.05, 'ci_hi': 0.30, 'n_participants': 4,
                     'stable': True},
                    {'cue_community': 2, 'to_community': 3, 'count': 3, 'selection_freq': 0.2,
                     'mean_delta_prog': None, 'ci_lo': None, 'ci_hi': None, 'n_participants': 2,
                     'stable': False}]
        name_rows = [{'community_id': 0, 'top_terms': ['breath', 'notice'], 'dominant_speaker': 'therapist'},
                     {'community_id': 1, 'top_terms': ['pain', 'back'], 'dominant_speaker': 'participant'}]
        with tempfile.TemporaryDirectory() as tmp:
            csv = COMM.write_dyadic_csv(routines, tmp)
            self.assertTrue(csv and os.path.isfile(csv))
            rep = COMM.write_dyadic_report(routines, name_rows, tmp)
            self.assertTrue(os.path.isfile(rep))
            txt = open(rep, encoding='utf-8').read()
            self.assertIn('DYADIC ROUTINES', txt)
            self.assertIn('STABLE ROUTINES', txt)


if __name__ == '__main__':
    unittest.main()
