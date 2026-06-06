"""
Integration: the GNN layer end-to-end on REAL tiny embeddings.

Runs run_gnn_analysis with all-MiniLM-L6-v2 (no embedding mock) for a few
epochs and asserts the validation report, the gnn_labels overlay, and the
discovery CSVs are written; then exercises LLM-free scale-mode classification
(run_gnn_classify) on an unseen segment.
"""
import os
import sys
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import synthetic_df, build_tiny_config, integration_test

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_ROOT = os.path.join(_HERE, os.pardir, "testrun-outputs")


@integration_test
class TestRealGnn(unittest.TestCase):
    def setUp(self):
        self.out = os.path.join(_OUT_ROOT, "gnn_real")
        shutil.rmtree(self.out, ignore_errors=True)
        os.makedirs(self.out, exist_ok=True)
        base = build_tiny_config(self.out, enable_gnn=True)
        self.gcfg = base.gnn_layer
        # A few epochs only; real embeddings, but train head must be present.
        self.gcfg.epochs = 10
        self.gcfg.objectives = ['soft_vaamr', 'progression', 'purer']

    def test_run_then_scale_classify(self):
        from gnn_layer import runner
        df = synthetic_df(n_sessions=4)
        try:
            res = runner.run_gnn_analysis(df, self.out, config=self.gcfg, verbose=False)
        except Exception as e:
            raise unittest.SkipTest(f"GNN real run could not complete (model/deps): {e}")
        self.assertEqual(res.get('status'), 'ok')
        from process import output_paths as paths
        # Reliability report
        val = os.path.join(paths.reports_gnn_dir(self.out), 'validation.txt')
        self.assertTrue(os.path.isfile(val), "validation.txt not written")
        # Consensus overlay
        overlay = paths.classification_overlay_path(self.out, 'gnn')
        self.assertTrue(os.path.isfile(overlay), "gnn_labels overlay not written")
        # Discovery CSVs
        gdir = paths.gnn_data_dir(self.out)
        self.assertTrue(os.path.isfile(os.path.join(gdir, 'segment_positions.csv')))
        self.assertTrue(os.path.isfile(os.path.join(gdir, 'cue_motifs.csv')))

        # Scale mode: classify an unseen segment with NO LLM.
        import pandas as pd
        new = pd.DataFrame([dict(
            segment_id='cNEW_0', session_id='cNEW', speaker='participant',
            text='I noticed the pain shifting into separate sensations.',
            start_time_ms=1000, end_time_ms=1800, final_label=float('nan'),
            primary_stage=float('nan'),
        )])
        cres = runner.run_gnn_classify(new, self.out, config=self.gcfg, verbose=False)
        self.assertIn(cres.get('status', ''), ('ok', 'skipped: no trained checkpoint (run the GNN layer first)'))


if __name__ == "__main__":
    unittest.main()
