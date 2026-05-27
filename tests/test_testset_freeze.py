"""Tests for flat numbered testset create/refresh logic in process/assembly/human_forms.py."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_tools.data_structures import Segment
from process._freeze import FrozenArtifactError
from process.assembly.human_forms import (
    create_frozen_testset,
    refresh_testset_answer_key,
    generate_or_refresh_validation_testsets,
    _collect_used_segment_ids,
    _parse_worksheet_items,
)
from process import output_paths as _paths


def _make_segment(segment_id, session_id='c1s1', cohort_id=1, speaker='participant',
                  text='A test segment.', segment_index=0, primary_stage=1):
    return Segment(
        segment_id=segment_id,
        trial_id='t1',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=cohort_id,
        session_variant='',
        segment_index=segment_index,
        start_time_ms=0,
        end_time_ms=3000,
        total_segments_in_session=5,
        speaker=speaker,
        text=text,
        word_count=3,
        speakers_in_segment=[speaker],
        session_file='',
        primary_stage=primary_stage,
        llm_confidence_primary=0.9,
        agreement_level='unanimous',
    )


def _make_pool(n=10):
    return [
        _make_segment(f'seg_{i:03d}', segment_index=i, text=f'Segment number {i}.')
        for i in range(n)
    ]


class TestCreateFrozenTestset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_flat_files(self):
        segs = _make_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=2, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)))
        self.assertEqual(human_path, _paths.testset_human_flat_path(self.tmpdir, 1))

    def test_no_directory_or_manifest_created(self):
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        ts_dir = _paths.testsets_dir(self.tmpdir)
        entries = os.listdir(ts_dir)
        self.assertFalse(any(os.path.isdir(os.path.join(ts_dir, e)) for e in entries),
                         "No subdirectories should be created")
        self.assertFalse(any('manifest' in e for e in entries),
                         "No manifest.json file should be created")
        self.assertFalse(any(e.endswith('.meta.json') for e in entries),
                         "Sidecar .meta.json must live under 02_meta, not the worksheet dir")

    def test_meta_sidecar_created(self):
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        meta_path = _paths.testset_meta_path(self.tmpdir, 1)
        self.assertTrue(os.path.isfile(meta_path), "Sidecar .meta.json must be written")
        self.assertIn(os.path.join('02_meta', 'testset_meta'), meta_path,
                      "Sidecar must be stored under 02_meta/testset_meta")
        import json
        with open(meta_path) as fh:
            meta = json.load(fh)
        self.assertIn('segments', meta)
        self.assertTrue(len(meta['segments']) > 0)
        first = meta['segments'][0]
        self.assertIn('session_id', first)
        self.assertIn('seg_num', first)
        self.assertIn('sha256', first)
        self.assertEqual(len(first['sha256']), 64, "SHA256 hex digest should be 64 chars")

    def test_sequential_numbering(self):
        segs = _make_pool(10)
        p1 = create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        p2 = create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        self.assertEqual(p1, _paths.testset_human_flat_path(self.tmpdir, 1))
        self.assertEqual(p2, _paths.testset_human_flat_path(self.tmpdir, 2))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))


class TestRefreshTestsetAnswerKey(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segs = _make_pool(10)
        create_frozen_testset(
            self.segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_refresh_updates_ai_key_only(self):
        import time
        human_path = _paths.testset_human_flat_path(self.tmpdir, 1)
        mtime_human_before = os.path.getmtime(human_path)
        time.sleep(0.05)

        segs_by_id = {s.segment_id: s for s in self.segs}
        ai_path = refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                             codebook_enabled=False)

        self.assertEqual(os.path.getmtime(human_path), mtime_human_before,
                         "Human worksheet must not be modified by refresh")
        self.assertTrue(os.path.isfile(ai_path))

    def test_missing_segment_raises(self):
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key({}, None, self.tmpdir, 1,
                                       codebook_enabled=False)

    def test_missing_worksheet_raises(self):
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key({}, None, self.tmpdir, 99,
                                       codebook_enabled=False)


class TestGenerateOrRefreshCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_testsets_on_first_call(self):
        segs = _make_pool(20)
        paths = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=2, fraction_per_set=0.2,
            random_seed=42, codebook_enabled=False,
        )
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f"Expected file: {p}")

    def test_creates_two_sequential_worksheets(self):
        segs = _make_pool(20)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=2, fraction_per_set=0.2,
            random_seed=42, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 2)))

    def test_refresh_only_does_not_create_new_files(self):
        # create_missing=False with no existing worksheets: nothing created
        segs = _make_pool(20)
        paths = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=2, fraction_per_set=0.2,
            random_seed=42, codebook_enabled=False, create_missing=False,
        )
        self.assertEqual(paths, [])
        ts_dir = _paths.testsets_dir(self.tmpdir)
        self.assertFalse(os.path.isdir(ts_dir) and bool(os.listdir(ts_dir)))

    def test_refresh_regenerates_ai_keys(self):
        import time
        segs = _make_pool(20)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=1, fraction_per_set=0.3,
            random_seed=42, codebook_enabled=False,
        )
        human_path = _paths.testset_human_flat_path(self.tmpdir, 1)
        ai_path = _paths.testset_ai_flat_path(self.tmpdir, 1)
        mtime_human = os.path.getmtime(human_path)
        time.sleep(0.05)

        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=1, fraction_per_set=0.3,
            random_seed=42, codebook_enabled=False, create_missing=False,
        )
        # Human worksheet untouched
        self.assertEqual(os.path.getmtime(human_path), mtime_human)
        # AI key refreshed
        self.assertTrue(os.path.isfile(ai_path))


class TestDeduplication(unittest.TestCase):
    """create_frozen_testset with exclude_segment_ids prevents overlap between testsets."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_with_dedup(self, segs, **kwargs):
        used_ids = _collect_used_segment_ids(self.tmpdir)
        return create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=kwargs.get('fraction_per_set', 0.3),
            random_seed=kwargs.get('random_seed', 42),
            codebook_enabled=False,
            exclude_segment_ids=used_ids,
        )

    def test_no_overlap_between_two_testsets(self):
        segs = _make_pool(20)
        self._create_with_dedup(segs)
        self._create_with_dedup(segs)

        items1 = set(_parse_worksheet_items(_paths.testset_human_flat_path(self.tmpdir, 1)))
        items2 = set(_parse_worksheet_items(_paths.testset_human_flat_path(self.tmpdir, 2)))
        overlap = items1 & items2
        self.assertEqual(overlap, set(), f"Testsets must not share segments; found: {overlap}")

    def test_no_overlap_with_three_testsets(self):
        segs = _make_pool(30)
        for _ in range(3):
            self._create_with_dedup(segs)

        items = [
            set(_parse_worksheet_items(_paths.testset_human_flat_path(self.tmpdir, n)))
            for n in (1, 2, 3)
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = items[i] & items[j]
                self.assertEqual(overlap, set(),
                                 f"Testsets #{i + 1} and #{j + 1} share segments: {overlap}")

    def test_empty_pool_raises_valueerror(self):
        segs = _make_pool(3)
        # First testset: take ALL segments from the pool (fraction=1.0)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=42, codebook_enabled=False,
        )
        # Now all segments are recorded as used; a second creation should fail
        used_ids = _collect_used_segment_ids(self.tmpdir)
        self.assertTrue(len(used_ids) > 0, "used_ids should be populated after first testset")
        with self.assertRaises(ValueError, msg="Should raise when all pool segments are used"):
            create_frozen_testset(
                segs, None, self.tmpdir,
                n_sets=1, set_index=1,
                fraction_per_set=1.0, random_seed=42, codebook_enabled=False,
                exclude_segment_ids=used_ids,
            )

    def test_collect_used_segment_ids_empty_dir(self):
        used = _collect_used_segment_ids(self.tmpdir)
        self.assertEqual(used, set())

    def test_collect_used_segment_ids_after_create(self):
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=42, codebook_enabled=False,
        )
        used = _collect_used_segment_ids(self.tmpdir)
        self.assertTrue(len(used) > 0)
        for session_id, seg_num in used:
            self.assertIsInstance(session_id, str)
            self.assertIsInstance(seg_num, int)
            self.assertGreaterEqual(seg_num, 1)


class TestHeaderSync(unittest.TestCase):
    """Headers update to reflect current total when new testsets are added."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _header_line(self, n):
        with open(_paths.testset_human_flat_path(self.tmpdir, n), encoding='utf-8') as fh:
            for line in fh:
                if 'VALIDATION TEST SET' in line:
                    return line.strip()
        return None

    def test_single_testset_says_1_of_1(self):
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        header = self._header_line(1)
        self.assertIn('1 of 1', header)

    def test_header_updates_to_1_of_2_after_refresh(self):
        segs = _make_pool(20)
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        create_frozen_testset(
            segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=0.3, random_seed=99, codebook_enabled=False,
        )
        segs_by_id = {s.segment_id: s for s in segs}
        refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                   codebook_enabled=False, n_total=2)
        header = self._header_line(1)
        self.assertIn('1 of 2', header, f"Expected '1 of 2' in header, got: {header}")

    def test_all_headers_updated_via_refresh_loop(self):
        segs = _make_pool(30)
        for i in range(3):
            create_frozen_testset(
                segs, None, self.tmpdir,
                n_sets=1, set_index=1,
                fraction_per_set=0.2, random_seed=42 + i, codebook_enabled=False,
            )
        segs_by_id = {s.segment_id: s for s in segs}
        for n in (1, 2, 3):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, n,
                                       codebook_enabled=False, n_total=3)
        for n in (1, 2, 3):
            header = self._header_line(n)
            self.assertIn(f'{n} of 3', header, f"Testset #{n} header wrong: {header}")


class TestContentChangeWarning(unittest.TestCase):
    """refresh_testset_answer_key hard-errors on segment drift unless force=True."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segs = _make_pool(10)
        # Use fraction=1.0 so ALL segments are in the testset; mutation is guaranteed to hit
        create_frozen_testset(
            self.segs, None, self.tmpdir,
            n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=42, codebook_enabled=False,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mutated_segs_by_id(self):
        """Return segments_by_id where one segment's text has changed."""
        mutated = []
        for i, s in enumerate(self.segs):
            if i == 0:
                mutated.append(_make_segment(
                    s.segment_id, session_id=s.session_id,
                    segment_index=s.segment_index, text='CHANGED TEXT for testing.'
                ))
            else:
                mutated.append(s)
        return {s.segment_id: s for s in mutated}

    def test_succeeds_silently_when_content_unchanged(self):
        import io, contextlib
        segs_by_id = {s.segment_id: s for s in self.segs}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                       codebook_enabled=False)
        self.assertNotIn('WARNING', buf.getvalue())

    def test_hard_error_when_content_changes_without_force(self):
        segs_by_id = self._mutated_segs_by_id()
        ai_path = _paths.testset_ai_flat_path(self.tmpdir, 1)
        with open(ai_path, encoding='utf-8') as fh:
            before = fh.read()
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                       codebook_enabled=False)
        # AI key must be left untouched when the refresh aborts.
        with open(ai_path, encoding='utf-8') as fh:
            self.assertEqual(before, fh.read())

    def test_force_regenerates_against_changed_text(self):
        import io, contextlib
        segs_by_id = self._mutated_segs_by_id()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                       codebook_enabled=False, force=True)
        self.assertIn('WARNING', buf.getvalue())
        with open(_paths.testset_ai_flat_path(self.tmpdir, 1), encoding='utf-8') as fh:
            ai_content = fh.read()
        self.assertIn('CHANGED TEXT for testing.', ai_content)

    def test_no_validation_if_no_meta_sidecar(self):
        """Gracefully skips content check when .meta.json is absent (legacy testsets)."""
        import io, contextlib
        # Remove the meta file to simulate a testset with no recorded SHA256.
        meta_path = _paths.testset_meta_path(self.tmpdir, 1)
        os.remove(meta_path)

        segs_by_id = self._mutated_segs_by_id()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                       codebook_enabled=False)
        self.assertNotIn('WARNING', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
