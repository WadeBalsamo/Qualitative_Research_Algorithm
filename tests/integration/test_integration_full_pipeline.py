"""
Integration: the whole pipeline on synthetic VTTs with real tiny models.

ingest (real all-MiniLM-L6-v2) -> VAAMR + PURER classify (real tiny Ollama)
-> assemble -> analyze (GNN enabled, few epochs). The "everything runs"
smoke test. Skips if Ollama is unavailable.
"""
import os
import sys
import shutil
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import build_tiny_config, integration_test
from tests.testhelpers.ollama_helper import ensure_ollama_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTDATA = os.path.join(_HERE, os.pardir, "testdata")
_OUT_ROOT = os.path.join(_HERE, os.pardir, "testrun-outputs")


@integration_test
class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.llm = ensure_ollama_model()  # SkipTest if Ollama/model absent
        self.out = os.path.join(_OUT_ROOT, "full_pipeline")
        shutil.rmtree(self.out, ignore_errors=True)
        os.makedirs(self.out, exist_ok=True)
        cfg = build_tiny_config(self.out, transcript_dir=_TESTDATA, enable_gnn=True)
        for sub in (cfg.theme_classification, cfg.purer_classification):
            sub.backend = self.llm.backend
            sub.model = self.llm.model
            sub.ollama_host = getattr(self.llm, "ollama_host", "127.0.0.1")
            sub.ollama_port = getattr(self.llm, "ollama_port", 11434)
            sub.n_runs = 1
            sub.temperature = 0.0
        cfg.gnn_layer.epochs = 8
        cfg.auto_analyze = True
        self.cfg = cfg

    def test_pipeline_produces_master_and_reports(self):
        from process.orchestrator import run_full_pipeline
        from constructs.registry import load
        fw = load('vaamr')
        df = run_full_pipeline(self.cfg, fw, codebook=None)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        # master_segments CSV export written (frozen segments + overlays live in qra.db;
        # the master dataset is exported only as CSV — no longer master_segments.jsonl)
        train_dir = os.path.join(self.out, "02_meta", "training_data")
        master = os.path.join(train_dir, "master_segments.csv")
        self.assertTrue(os.path.isfile(master), "master_segments.csv missing")

        # At least one participant got a VAAMR final_label
        if "final_label" in df.columns and "speaker" in df.columns:
            parts = df[df["speaker"] == "participant"]
            self.assertTrue(parts["final_label"].notna().any(),
                            "no participant received a VAAMR label")

        # 06_reports tree exists
        reports = os.path.join(self.out, "06_reports")
        self.assertTrue(os.path.isdir(reports), "06_reports not generated")


if __name__ == "__main__":
    unittest.main()
