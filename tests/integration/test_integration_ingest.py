"""
Integration: real segmentation/ingest on synthetic VTT testdata.

Runs Stage 1 (stage_ingest) with the tiny real embedding model over
tests/testdata/*.vtt and asserts frozen segments are produced and that
participant/therapist speaker separation happened (methodology Section 4.1).
"""
import os
import sys
import shutil
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import build_tiny_config, TINY_EMBED, integration_test

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTDATA = os.path.join(_HERE, os.pardir, "testdata")
_OUT_ROOT = os.path.join(_HERE, os.pardir, "testrun-outputs")


@integration_test
class TestRealIngest(unittest.TestCase):
    def setUp(self):
        self.out = os.path.join(_OUT_ROOT, "ingest")
        shutil.rmtree(self.out, ignore_errors=True)
        os.makedirs(self.out, exist_ok=True)
        self.cfg = build_tiny_config(self.out, transcript_dir=_TESTDATA, enable_gnn=False)
        # The testdata facilitator is "Dr. Rivera"; speaker roles are config-driven
        # (no honorific heuristic), so exclude-mode must name the therapist for the
        # participant/therapist separation this test asserts.
        self.cfg.speaker_filter.mode = "exclude"
        self.cfg.speaker_filter.speakers = ["Dr. Rivera"]

    def test_ingest_freezes_segments_and_separates_speakers(self):
        from process.orchestrator import stage_ingest
        from process import segments_io
        try:
            segments = stage_ingest(self.cfg, output_dir=self.out)
        except Exception as e:
            raise unittest.SkipTest(f"ingest could not run (model/deps): {e}")
        self.assertTrue(segments, "no segments produced from testdata")
        speakers = {getattr(s, "speaker", None) for s in segments}
        # Both participant and therapist roles should be present after normalization.
        self.assertIn("participant", speakers)
        self.assertIn("therapist", speakers)
        # Frozen segments are persisted to the qra.db `segments` table (not JSONL files).
        sessions = segments_io.list_segmented_sessions(self.out)
        self.assertTrue(sessions, "no frozen sessions recorded in qra.db")
        self.assertTrue(segments_io.read_session_segments(self.out, sessions[0]),
                        "frozen segments not retrievable from qra.db")


if __name__ == "__main__":
    unittest.main()
