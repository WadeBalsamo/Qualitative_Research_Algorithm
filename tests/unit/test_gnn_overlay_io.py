"""
tests/unit/test_gnn_overlay_io.py
----------------------------------
Unit tests for the 'gnn' overlay in process/classifications_io.py.

The existing test_classifications_io.py covers theme/purer/codebook/cv overlays
and manifest operations.  test_gnn_consensus.py covers roundtrip/merge/registry.

THIS FILE covers the gaps specific to the GNN overlay:
  - GNN_OVERLAY_FIELDS constant contents
  - OVERLAY_KEYS and OVERLAY_FILENAMES include 'gnn'
  - write_gnn_overlay file path and JSONL structure
  - apply_gnn_overlay attaches all 5 GNN fields to Segment objects
  - apply_overlays (multi-key) includes and applies the 'gnn' key
  - Provenance manifest updated for the 'gnn' key and siblings not clobbered
  - Empty overlay roundtrip (zero segments)
  - Partial-match: only segments whose IDs appear in overlay are updated
  - Sorted-record guarantee for GNN overlay
  - gnn overlay overwrite semantics (force=True via write_frozen)
  - apply_gnn_overlay on missing file returns 0 (no crash)
  - apply_overlays silently skips absent overlay files
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import classifications_io as cio


def _run_dir():
    """Create a temp dir with the 02_meta/classifications sub-tree."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, '02_meta', 'classifications'), exist_ok=True)
    return d


def _gnn_seg(segment_id, vaamr_pred=None, vaamr_conf=None,
             purer_pred=None, purer_conf=None, label_source=None, **kwargs):
    """Build a Segment with GNN fields set."""
    seg = Segment(segment_id=segment_id, **kwargs)
    seg.gnn_vaamr_pred = vaamr_pred
    seg.gnn_vaamr_conf = vaamr_conf
    seg.gnn_purer_pred = purer_pred
    seg.gnn_purer_conf = purer_conf
    seg.gnn_label_source = label_source
    return seg


# ---------------------------------------------------------------------------
# Constants / registry
# ---------------------------------------------------------------------------

class TestGnnConstants(unittest.TestCase):

    def test_overlay_keys_includes_gnn(self):
        self.assertIn('gnn', cio.OVERLAY_KEYS)

    def test_overlay_filenames_maps_gnn_to_gnn_labels_jsonl(self):
        self.assertEqual(cio.OVERLAY_FILENAMES.get('gnn'), 'gnn_labels.jsonl')

    def test_gnn_overlay_fields_tuple_has_expected_fields(self):
        expected = {
            'gnn_vaamr_pred', 'gnn_vaamr_conf', 'gnn_vaamr_abstain',
            'gnn_purer_pred', 'gnn_purer_conf', 'gnn_purer_abstain',
            'gnn_label_source',
        }
        self.assertEqual(set(cio.GNN_OVERLAY_FIELDS), expected)

    def test_gnn_overlay_fields_are_real_segment_attributes(self):
        """Every GNN_OVERLAY_FIELD must exist as a dataclass field on Segment."""
        seg = Segment(segment_id='x')
        for field in cio.GNN_OVERLAY_FIELDS:
            self.assertTrue(hasattr(seg, field),
                            f'Segment is missing field {field!r} listed in GNN_OVERLAY_FIELDS')


# ---------------------------------------------------------------------------
# write_gnn_overlay — file structure
# ---------------------------------------------------------------------------

class TestWriteGnnOverlay(unittest.TestCase):

    def setUp(self):
        self.run_dir = _run_dir()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_write_creates_file_at_expected_path(self):
        seg = _gnn_seg('p1', vaamr_pred=2, vaamr_conf=0.9, label_source='gnn_trained')
        path = cio.write_gnn_overlay(self.run_dir, [seg])
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith('gnn_labels.jsonl'))
        self.assertIn('02_meta', path)
        self.assertIn('classifications', path)

    def test_write_produces_valid_jsonl(self):
        segs = [
            _gnn_seg('p1', vaamr_pred=1, vaamr_conf=0.8, label_source='gnn_trained'),
            _gnn_seg('t2', purer_pred=3, purer_conf=0.7, label_source='gnn_trained'),
        ]
        cio.write_gnn_overlay(self.run_dir, segs)
        path = cio.overlay_path(self.run_dir, 'gnn')
        with open(path, encoding='utf-8') as fh:
            lines = [ln for ln in fh if ln.strip()]
        self.assertEqual(len(lines), 2)
        for ln in lines:
            rec = json.loads(ln)
            self.assertIn('segment_id', rec)
            # every GNN_OVERLAY_FIELD present as key
            for f in cio.GNN_OVERLAY_FIELDS:
                self.assertIn(f, rec, f'GNN overlay record missing field {f!r}')

    def test_write_empty_overlay_creates_empty_file(self):
        cio.write_gnn_overlay(self.run_dir, [])
        path = cio.overlay_path(self.run_dir, 'gnn')
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding='utf-8') as fh:
            lines = [ln for ln in fh if ln.strip()]
        self.assertEqual(len(lines), 0)

    def test_write_records_sorted_by_segment_id(self):
        segs = [
            _gnn_seg('seg_003', vaamr_pred=0),
            _gnn_seg('seg_001', vaamr_pred=1),
            _gnn_seg('seg_002', vaamr_pred=2),
        ]
        cio.write_gnn_overlay(self.run_dir, segs)
        path = cio.overlay_path(self.run_dir, 'gnn')
        with open(path, encoding='utf-8') as fh:
            ids = [json.loads(ln)['segment_id'] for ln in fh if ln.strip()]
        self.assertEqual(ids, sorted(ids))

    def test_overlay_can_be_overwritten(self):
        """write_gnn_overlay uses force=True — should not raise on second call."""
        seg = _gnn_seg('p1', vaamr_pred=1)
        cio.write_gnn_overlay(self.run_dir, [seg])
        seg.gnn_vaamr_pred = 4
        cio.write_gnn_overlay(self.run_dir, [seg])  # should not raise
        by_id = {'p1': Segment(segment_id='p1')}
        cio.apply_gnn_overlay(self.run_dir, by_id)
        self.assertEqual(by_id['p1'].gnn_vaamr_pred, 4)


# ---------------------------------------------------------------------------
# apply_gnn_overlay — field restoration
# ---------------------------------------------------------------------------

class TestApplyGnnOverlay(unittest.TestCase):

    def setUp(self):
        self.run_dir = _run_dir()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_all_five_gnn_fields_restored(self):
        """Every GNN_OVERLAY_FIELD is applied back to the Segment."""
        seg = _gnn_seg('p1', vaamr_pred=3, vaamr_conf=0.88,
                       purer_pred=2, purer_conf=0.71,
                       label_source='gnn_trained')
        cio.write_gnn_overlay(self.run_dir, [seg])

        blank = Segment(segment_id='p1')
        n = cio.apply_gnn_overlay(self.run_dir, {'p1': blank})

        self.assertEqual(n, 1)
        self.assertEqual(blank.gnn_vaamr_pred, 3)
        self.assertAlmostEqual(blank.gnn_vaamr_conf, 0.88)
        self.assertEqual(blank.gnn_purer_pred, 2)
        self.assertAlmostEqual(blank.gnn_purer_conf, 0.71)
        self.assertEqual(blank.gnn_label_source, 'gnn_trained')

    def test_apply_none_fields_preserved(self):
        """Segments with None GNN fields write/read None without error."""
        seg = _gnn_seg('p2', vaamr_pred=None, vaamr_conf=None,
                       purer_pred=None, purer_conf=None, label_source=None)
        cio.write_gnn_overlay(self.run_dir, [seg])

        blank = Segment(segment_id='p2')
        n = cio.apply_gnn_overlay(self.run_dir, {'p2': blank})
        self.assertEqual(n, 1)
        self.assertIsNone(blank.gnn_vaamr_pred)
        self.assertIsNone(blank.gnn_label_source)

    def test_apply_missing_file_returns_zero(self):
        """No overlay file → apply returns 0 and leaves segment untouched."""
        blank = Segment(segment_id='p1')
        n = cio.apply_gnn_overlay(self.run_dir, {'p1': blank})
        self.assertEqual(n, 0)
        self.assertIsNone(blank.gnn_vaamr_pred)

    def test_only_matching_ids_updated(self):
        """Segments whose IDs are absent from the overlay are left untouched."""
        seg = _gnn_seg('p_known', vaamr_pred=1)
        cio.write_gnn_overlay(self.run_dir, [seg])

        known = Segment(segment_id='p_known')
        unknown = Segment(segment_id='p_unknown')
        n = cio.apply_gnn_overlay(self.run_dir, {
            'p_known': known, 'p_unknown': unknown,
        })
        self.assertEqual(n, 1)
        self.assertEqual(known.gnn_vaamr_pred, 1)
        self.assertIsNone(unknown.gnn_vaamr_pred)

    def test_apply_empty_overlay_returns_zero(self):
        """Empty overlay file → apply returns 0."""
        cio.write_gnn_overlay(self.run_dir, [])
        blank = Segment(segment_id='p1')
        n = cio.apply_gnn_overlay(self.run_dir, {'p1': blank})
        self.assertEqual(n, 0)

    def test_apply_multiple_segments_returns_correct_count(self):
        segs = [_gnn_seg(f'seg_{i}', vaamr_pred=i % 5) for i in range(5)]
        cio.write_gnn_overlay(self.run_dir, segs)
        by_id = {seg.segment_id: Segment(segment_id=seg.segment_id) for seg in segs}
        n = cio.apply_gnn_overlay(self.run_dir, by_id)
        self.assertEqual(n, 5)
        for i in range(5):
            self.assertEqual(by_id[f'seg_{i}'].gnn_vaamr_pred, i % 5)


# ---------------------------------------------------------------------------
# apply_overlays — multi-key dispatch includes 'gnn'
# ---------------------------------------------------------------------------

class TestApplyOverlaysMultiKey(unittest.TestCase):

    def setUp(self):
        self.run_dir = _run_dir()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_apply_overlays_gnn_key_in_result(self):
        """apply_overlays returns a dict that includes a 'gnn' entry."""
        result = cio.apply_overlays(self.run_dir, {})
        self.assertIn('gnn', result)

    def test_apply_overlays_with_only_gnn_written(self):
        """apply_overlays applies GNN fields and silently skips absent overlays."""
        seg = _gnn_seg('p1', vaamr_pred=2, label_source='gnn_trained')
        cio.write_gnn_overlay(self.run_dir, [seg])

        blank = Segment(segment_id='p1')
        result = cio.apply_overlays(self.run_dir, {'p1': blank})

        # gnn overlay applied
        self.assertEqual(result['gnn'], 1)
        self.assertEqual(blank.gnn_vaamr_pred, 2)
        # other keys absent → count 0 (no crash)
        for key in ('theme', 'purer', 'codebook'):
            self.assertEqual(result[key], 0)

    def test_apply_overlays_with_subset_keys(self):
        """apply_overlays respects caller-supplied keys tuple."""
        seg = _gnn_seg('p1', vaamr_pred=0)
        cio.write_gnn_overlay(self.run_dir, [seg])
        blank = Segment(segment_id='p1')
        result = cio.apply_overlays(self.run_dir, {'p1': blank}, keys=('gnn',))
        self.assertIn('gnn', result)
        self.assertNotIn('theme', result)
        self.assertEqual(result['gnn'], 1)

    def test_apply_overlays_all_absent_returns_zeros(self):
        """No overlay files at all → all counts are 0."""
        blank = Segment(segment_id='p1')
        result = cio.apply_overlays(self.run_dir, {'p1': blank})
        self.assertTrue(all(v == 0 for v in result.values()),
                        f'expected all zeros, got {result}')


# ---------------------------------------------------------------------------
# Manifest — gnn key
# ---------------------------------------------------------------------------

class TestManifestGnnKey(unittest.TestCase):

    def setUp(self):
        self.run_dir = _run_dir()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_manifest_records_gnn_entry(self):
        cio.update_classification_manifest(
            self.run_dir,
            key='gnn',
            entry={'model': 'gnn_v1', 'n_segments': 10, 'epochs': 50},
        )
        manifest = cio.read_classification_manifest(self.run_dir)
        self.assertIsNotNone(manifest)
        self.assertIn('gnn', manifest)
        self.assertEqual(manifest['gnn']['model'], 'gnn_v1')
        self.assertIn('completed_at', manifest['gnn'])

    def test_gnn_manifest_does_not_clobber_theme(self):
        cio.update_classification_manifest(
            self.run_dir, key='theme', entry={'model': 'llm_a'},
        )
        cio.update_classification_manifest(
            self.run_dir, key='gnn', entry={'model': 'gnn_v1'},
        )
        manifest = cio.read_classification_manifest(self.run_dir)
        self.assertEqual(manifest['theme']['model'], 'llm_a')
        self.assertEqual(manifest['gnn']['model'], 'gnn_v1')

    def test_gnn_manifest_overwritten_on_second_call(self):
        cio.update_classification_manifest(
            self.run_dir, key='gnn', entry={'model': 'old'},
        )
        cio.update_classification_manifest(
            self.run_dir, key='gnn', entry={'model': 'new'},
        )
        manifest = cio.read_classification_manifest(self.run_dir)
        self.assertEqual(manifest['gnn']['model'], 'new')

    def test_gnn_and_purer_manifest_together(self):
        """Both gnn and purer keys can coexist in the manifest."""
        cio.update_classification_manifest(
            self.run_dir, key='purer', entry={'model': 'purer_model'},
        )
        cio.update_classification_manifest(
            self.run_dir, key='gnn', entry={'model': 'gnn_model', 'epochs': 100},
        )
        manifest = cio.read_classification_manifest(self.run_dir)
        self.assertIn('purer', manifest)
        self.assertIn('gnn', manifest)
        self.assertEqual(manifest['purer']['model'], 'purer_model')
        self.assertEqual(manifest['gnn']['epochs'], 100)


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------

class TestMergeGnnOverlay(unittest.TestCase):

    def setUp(self):
        self.run_dir = _run_dir()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_merge_into_empty_overlay(self):
        """merge_gnn_overlay on a missing file acts like write."""
        seg = _gnn_seg('p1', vaamr_pred=2, label_source='gnn_trained')
        cio.merge_gnn_overlay(self.run_dir, [seg])
        by_id = {'p1': Segment(segment_id='p1')}
        cio.apply_gnn_overlay(self.run_dir, by_id)
        self.assertEqual(by_id['p1'].gnn_vaamr_pred, 2)

    def test_merge_upserts_existing_row(self):
        """merge replaces a row with matching segment_id."""
        seg = _gnn_seg('p1', vaamr_pred=1)
        cio.write_gnn_overlay(self.run_dir, [seg])
        seg.gnn_vaamr_pred = 4
        cio.merge_gnn_overlay(self.run_dir, [seg])
        by_id = {'p1': Segment(segment_id='p1')}
        cio.apply_gnn_overlay(self.run_dir, by_id)
        self.assertEqual(by_id['p1'].gnn_vaamr_pred, 4)

    def test_merge_preserves_non_updated_rows(self):
        """merge only replaces rows specified; untouched rows survive."""
        seg_a = _gnn_seg('pA', vaamr_pred=0)
        seg_b = _gnn_seg('pB', vaamr_pred=1)
        cio.write_gnn_overlay(self.run_dir, [seg_a, seg_b])
        seg_a.gnn_vaamr_pred = 3
        cio.merge_gnn_overlay(self.run_dir, [seg_a])
        by_id = {'pA': Segment(segment_id='pA'), 'pB': Segment(segment_id='pB')}
        cio.apply_gnn_overlay(self.run_dir, by_id)
        self.assertEqual(by_id['pA'].gnn_vaamr_pred, 3)
        self.assertEqual(by_id['pB'].gnn_vaamr_pred, 1)  # untouched

    def test_merged_file_sorted_by_segment_id(self):
        """Post-merge JSONL remains sorted by segment_id."""
        cio.write_gnn_overlay(self.run_dir, [_gnn_seg('seg_003', vaamr_pred=0)])
        cio.merge_gnn_overlay(self.run_dir, [_gnn_seg('seg_001', vaamr_pred=1)])
        path = cio.overlay_path(self.run_dir, 'gnn')
        with open(path, encoding='utf-8') as fh:
            ids = [json.loads(ln)['segment_id'] for ln in fh if ln.strip()]
        self.assertEqual(ids, sorted(ids))


if __name__ == '__main__':
    unittest.main()
