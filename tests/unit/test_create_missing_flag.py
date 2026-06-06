"""
Tests for create_missing flag on Phase 2 coordinators (Phase 3).

generate_or_refresh_validation_testsets and
generate_or_refresh_content_validity_testsets both gain create_missing=False
so that qra assemble can refresh without creating new testsets.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment
from process import segments_io


def _make_segment(segment_id, session_id='c1s1', primary=2, **kwargs):
    seg = Segment(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=0,
        start_time_ms=0,
        end_time_ms=5000,
        total_segments_in_session=10,
        speaker='participant',
        text='This is a test segment about chronic pain.',
        word_count=8,
        **kwargs,
    )
    seg.primary_stage = primary
    seg.agreement_level = 'unanimous'
    seg.agreement_fraction = 1.0
    return seg


class TestGenerateOrRefreshValidationTestsetsCreateMissing(unittest.TestCase):
    """generate_or_refresh_validation_testsets with create_missing=False."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from theme_framework.vaamr import get_vaamr_framework
        self.framework = get_vaamr_framework()

        # Create 20 labeled segments (enough for a testset)
        self.segments = [
            _make_segment(f'seg_{i:03d}', primary=(i % 5))
            for i in range(20)
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_missing_false_skips_nonexistent_testsets(self):
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        # With create_missing=False: no testset directory → nothing created
        dirs = generate_or_refresh_validation_testsets(
            self.segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg,
            create_missing=False,
        )
        self.assertEqual(dirs, [])

    def test_create_missing_true_creates_testset(self):
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        dirs = generate_or_refresh_validation_testsets(
            self.segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg,
            create_missing=True,
        )
        self.assertGreater(len(dirs), 0)

    def test_create_missing_false_refreshes_existing(self):
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        # Create first
        dirs1 = generate_or_refresh_validation_testsets(
            self.segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg, create_missing=True,
        )
        self.assertGreater(len(dirs1), 0)

        # Refresh (create_missing=False) — should succeed because it exists
        dirs2 = generate_or_refresh_validation_testsets(
            self.segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg, create_missing=False,
        )
        self.assertGreater(len(dirs2), 0)


class TestGenerateOrRefreshCVTestsetsCreateMissing(unittest.TestCase):
    """generate_or_refresh_content_validity_testsets with create_missing=False."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_missing_false_skips_nonexistent_cv_testsets(self):
        from process.assembly import generate_or_refresh_content_validity_testsets
        from theme_framework.vaamr import get_vaamr_framework
        from process.config import ContentValidityConfig, ContentValiditySpec

        framework = get_vaamr_framework()
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
        )

        # With create_missing=False: directory doesn't exist → nothing created
        dirs = generate_or_refresh_content_validity_testsets(
            self.tmpdir,
            cv_config=cv_cfg,
            framework_vaamr=framework,
            framework_purer=None,
            theme_classification_cfg=None,
            create_missing=False,
        )
        self.assertEqual(dirs, [])

    def test_create_missing_true_creates_cv_testset(self):
        from process.assembly import generate_or_refresh_content_validity_testsets
        from theme_framework.vaamr import get_vaamr_framework
        from process.config import ContentValidityConfig, ContentValiditySpec

        framework = get_vaamr_framework()
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
        )

        dirs = generate_or_refresh_content_validity_testsets(
            self.tmpdir,
            cv_config=cv_cfg,
            framework_vaamr=framework,
            framework_purer=None,
            theme_classification_cfg=None,
            create_missing=True,
        )
        self.assertGreater(len(dirs), 0)


if __name__ == '__main__':
    unittest.main()
