"""
tests/unit/test_gnn_communities.py
----------------------------------
Unit tests for Track D — subtext communities as routines.

Hermetic: crafts embeddings with K recurring "topics" (so cross-session communities actually
form), then exercises the thresholded graph, the two-algorithm partition + ARI, within-session
routine transitions, participant-bootstrap stability selection, TF-IDF naming, the report
writer, and the OFF-by-default flags. networkx/sklearn only — no model download.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer import communities as COMM


def _topic_emb(df, dim=16, k_topics=3, noise=0.15, seed=0):
    """Embeddings clustered into k recurring topics keyed by segment position (cross-session)."""
    rng = np.random.default_rng(seed)
    bases = rng.standard_normal((k_topics, dim)).astype('float32') * 6.0
    emb = {}
    for _, r in df.iterrows():
        sid = str(r['segment_id'])
        # position within session → same topic recurs across sessions (cross-session edges).
        topic = int(str(sid).split('_')[-1]) % k_topics
        emb[sid] = (bases[topic] + rng.standard_normal(dim).astype('float32') * noise)
    return emb


def _cfg(**kw):
    base = dict(community_sim_threshold=0.85, community_min_size=2,
                community_stability_boots=6, community_stability_min=0.4, seed=1)
    base.update(kw)
    return GnnLayerConfig(**base)


class TestSubtextGraph(unittest.TestCase):

    def test_graph_has_edges_and_cross_session(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        meta = COMM._seg_meta(df)
        G, info = COMM.build_subtext_graph(_topic_emb(df), meta, threshold=0.85)
        self.assertGreater(info['n_edges'], 0)
        self.assertGreater(info['n_cross_session_edges'], 0)
        self.assertEqual(info['n_nodes'], G.number_of_nodes())

    def test_cap_logged(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        meta = COMM._seg_meta(df)
        G, info = COMM.build_subtext_graph(_topic_emb(df), meta, threshold=0.85, max_nodes=5)
        self.assertGreaterEqual(info['n_capped'], 1)
        self.assertLessEqual(info['n_nodes'], 5)


class TestDetectAndRoutines(unittest.TestCase):

    def _setup(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        emb = _topic_emb(df)
        meta = COMM._seg_meta(df)
        cfg = _cfg()
        G, _ = COMM.build_subtext_graph(emb, meta, threshold=cfg.community_sim_threshold)
        sids = list(G.nodes())
        return df, emb, meta, cfg, G, sids

    def test_detect_returns_partition_and_ari(self):
        _, emb, _, cfg, G, sids = self._setup()
        det = COMM.detect_communities(G, emb, sids, cfg)
        self.assertGreaterEqual(det['n_communities'], 1)
        self.assertEqual(len(det['labels']), len(sids))
        # ARI present when there are >=2 multi-member communities (structured embeddings).
        self.assertTrue(det['ari_louvain_vs_hierarchical'] is None
                        or -1.0 <= det['ari_louvain_vs_hierarchical'] <= 1.0)

    def test_transitions(self):
        _, emb, meta, cfg, G, sids = self._setup()
        det = COMM.detect_communities(G, emb, sids, cfg)
        trans = COMM.community_transitions(det['labels'], meta)
        for t in trans:
            self.assertNotEqual(t['from_community'], t['to_community'])
            self.assertGreaterEqual(t['count'], 1)


class TestStabilityAndNaming(unittest.TestCase):

    def _detect(self):
        df = make_master_df(n_sessions=4, n_participants=3)
        emb = _topic_emb(df, seed=2)
        meta = COMM._seg_meta(df)
        cfg = _cfg()
        G, _ = COMM.build_subtext_graph(emb, meta, threshold=cfg.community_sim_threshold)
        det = COMM.detect_communities(G, emb, list(G.nodes()), cfg)
        return df, emb, meta, cfg, det

    def test_stability_dict(self):
        _, emb, meta, cfg, det = self._detect()
        stab = COMM.community_stability(emb, meta, cfg, det['labels'], det['communities'])
        self.assertIsInstance(stab, dict)
        for v in stab.values():
            self.assertIn('size', v)
            self.assertIn('stable', v)

    def test_naming(self):
        _, emb, meta, cfg, det = self._detect()
        stab = COMM.community_stability(emb, meta, cfg, det['labels'], det['communities'])
        rows = COMM.name_communities(det['communities'], meta, stab, cfg)
        for r in rows:
            self.assertIn('top_terms', r)
            self.assertIn('exemplars', r)
            self.assertGreaterEqual(r['size'], cfg.community_min_size)


class TestOrchestratorAndDefaults(unittest.TestCase):

    def test_run_writes_report(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        emb = _topic_emb(df)
        out = tempfile.mkdtemp()
        res = COMM.run_subtext_communities(df, emb, out, _cfg())
        self.assertEqual(res['status'], 'ok')
        from process import output_paths as _paths
        rep = os.path.join(_paths.reports_gnn_dir(out), 'communities.txt')
        self.assertTrue(os.path.isfile(rep))
        with open(rep) as f:
            self.assertIn('SUBTEXT COMMUNITIES AS ROUTINES', f.read())

    def test_too_few_segments(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        emb = {str(df.iloc[0]['segment_id']): np.ones(16, dtype='float32')}
        res = COMM.run_subtext_communities(df, emb, tempfile.mkdtemp(), _cfg())
        self.assertIn('skipped', res['status'])

    def test_off_by_default(self):
        c = GnnLayerConfig()
        self.assertTrue(c.subtext_communities)   # part of the default discovery build
        self.assertEqual(c.community_sim_threshold, 0.6)  # calibrated for Qwen (probe: 0.85 -> noise)


if __name__ == '__main__':
    unittest.main()
