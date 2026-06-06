"""Tests for process/legacy_migration.py."""
import json
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process import legacy_migration, output_paths as _paths


def _write_master_jsonl(run_dir, rows):
    """Write a minimal master_segments.jsonl for migration tests."""
    ms_dir = _paths.master_segments_dir(run_dir)
    os.makedirs(ms_dir, exist_ok=True)
    path = os.path.join(ms_dir, 'master_segments.jsonl')
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return path


def _make_row(segment_id, session_id, segment_index=0, cohort_id=1, text='Test text.'):
    return {
        'segment_id': segment_id,
        'trial_id': 'trial_A',
        'participant_id': 'participant_1',
        'session_id': session_id,
        'session_number': 1,
        'cohort_id': cohort_id,
        'session_variant': '',
        'segment_index': segment_index,
        'start_time_ms': 0,
        'end_time_ms': 3000,
        'total_segments_in_session': 2,
        'speaker': 'participant',
        'text': text,
        'word_count': 2,
        'session_file': '',
        'primary_stage': 1,
        'secondary_stage': None,
        'llm_confidence_primary': 0.9,
        'llm_confidence_secondary': None,
        'agreement_level': 'unanimous',
        'agreement_fraction': 1.0,
        'needs_review': False,
        'label_status': 'llm_only',
    }


class TestIsLegacyProject(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detects_legacy(self):
        _write_master_jsonl(self.tmpdir, [_make_row('seg_1', 'c1s1')])
        self.assertTrue(legacy_migration.is_legacy_project(self.tmpdir))

    def test_not_legacy_when_segmented_exists(self):
        _write_master_jsonl(self.tmpdir, [_make_row('seg_1', 'c1s1')])
        seg_dir = _paths.segmented_session_dir(self.tmpdir, 'c1s1')
        os.makedirs(seg_dir)
        open(os.path.join(seg_dir, 'segments.jsonl'), 'w').close()
        self.assertFalse(legacy_migration.is_legacy_project(self.tmpdir))

    def test_not_legacy_when_no_master(self):
        self.assertFalse(legacy_migration.is_legacy_project(self.tmpdir))


class TestMigrateLegacySegments(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _write_master_jsonl(self.tmpdir, [
            _make_row('seg_1', 'c1s1', segment_index=0),
            _make_row('seg_2', 'c1s1', segment_index=1),
            _make_row('seg_3', 'c1s2', segment_index=0),
        ])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_per_session_files(self):
        from process import segments_io as sio
        n = legacy_migration.migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n, 2)
        sessions = sio.list_segmented_sessions(self.tmpdir)
        self.assertIn('c1s1', sessions)
        self.assertIn('c1s2', sessions)
        self.assertTrue(sio.read_session_segments(self.tmpdir, 'c1s1'))
        self.assertTrue(sio.read_session_segments(self.tmpdir, 'c1s2'))

    def test_meta_has_legacy_hash(self):
        from process import segments_io as sio
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        # Legacy-migrated sessions carry the 'legacy-pre-modular' params_hash
        # sentinel, so is_segmentation_fresh returns True regardless of the
        # current params hash (no forced re-segmentation).
        self.assertTrue(sio.is_segmentation_fresh(self.tmpdir, 'c1s1', 'x'))

    def test_skips_already_migrated(self):
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        n = legacy_migration.migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n, 0)

    def test_segments_have_correct_ids(self):
        from process import segments_io
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        segs = segments_io.read_session_segments(self.tmpdir, 'c1s1')
        self.assertEqual({s.segment_id for s in segs}, {'seg_1', 'seg_2'})


class TestIsV25Layout(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detects_v25_by_diarized_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts', 'diarized'))
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_coded_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts', 'coded'))
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_flat_cv_file(self):
        val_dir = os.path.join(self.tmpdir, '04_validation')
        os.makedirs(val_dir)
        open(os.path.join(val_dir, 'content_validity_test_set.jsonl'), 'w').close()
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_flat_human_classification(self):
        val_dir = os.path.join(self.tmpdir, '04_validation')
        os.makedirs(val_dir)
        open(os.path.join(val_dir, 'human_classification_c1s1.txt'), 'w').close()
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_not_v25_for_fresh_project(self):
        self.assertFalse(legacy_migration.is_v25_layout(self.tmpdir))

    def test_not_v25_for_v3_with_inputs_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts_inputs'))
        os.makedirs(os.path.join(self.tmpdir, '04_validation', 'full_transcripts'))
        self.assertFalse(legacy_migration.is_v25_layout(self.tmpdir))


class TestMigrateV25ToV3(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dir(self, *parts):
        d = os.path.join(self.tmpdir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _touch(self, *parts):
        path = os.path.join(self.tmpdir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        return path

    def test_moves_diarized_to_inputs(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts_inputs', 'c1s1.vtt')))
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts', 'diarized', 'c1s1.vtt')))

    def test_removes_empty_diarized_dir(self):
        self._make_dir('01_transcripts', 'diarized')
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isdir(os.path.join(self.tmpdir, '01_transcripts', 'diarized')))

    def test_moves_coded_transcripts_to_full_transcripts(self):
        self._touch('01_transcripts', 'coded', 'coded_transcript_c1s1.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        full_tx = os.path.join(self.tmpdir, '04_validation', 'full_transcripts')
        self.assertTrue(os.path.isfile(os.path.join(full_tx, 'coded_transcript_c1s1.txt')))
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts', 'coded', 'coded_transcript_c1s1.txt')))

    def test_moves_human_classification_to_full_transcripts(self):
        val_dir = self._make_dir('04_validation')
        self._touch('04_validation', 'human_classification_c1s1.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        full_tx = os.path.join(self.tmpdir, '04_validation', 'full_transcripts')
        self.assertTrue(os.path.isfile(os.path.join(full_tx, 'human_classification_c1s1.txt')))
        self.assertFalse(os.path.isfile(os.path.join(val_dir, 'human_classification_c1s1.txt')))

    def test_moves_cv_files_to_content_validity(self):
        for fname in ['content_validity_test_set.jsonl', 'content_validity_human_worksheet.txt',
                      'content_validity_definition_key.txt', 'content_validity_answer_key.txt']:
            self._touch('04_validation', fname)
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        cv_dir = os.path.join(self.tmpdir, '04_validation', 'content_validity')
        for fname in ['content_validity_test_set.jsonl', 'content_validity_human_worksheet.txt',
                      'content_validity_definition_key.txt', 'content_validity_answer_key.txt']:
            self.assertTrue(os.path.isfile(os.path.join(cv_dir, fname)), f"Missing {fname}")
            self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '04_validation', fname)))

    def test_removes_worksheetN_base_files(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        self._touch('04_validation', 'testsets', 'worksheet1_base.txt')
        self._touch('04_validation', 'testsets', 'worksheet2_base.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isfile(os.path.join(ts_dir, 'worksheet1_base.txt')))
        self.assertFalse(os.path.isfile(os.path.join(ts_dir, 'worksheet2_base.txt')))

    def test_removes_folder_based_testset_dirs(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        legacy_dir = self._make_dir('04_validation', 'testsets', 'vaamr_testset_1')
        with open(os.path.join(legacy_dir, 'manifest.json'), 'w') as f:
            json.dump({'kind': 'vaamr'}, f)
        with open(os.path.join(legacy_dir, 'human_worksheet.txt'), 'w') as f:
            f.write('worksheet content')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isdir(legacy_dir))

    def test_does_not_touch_flat_frozen_worksheets(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        ws_path = os.path.join(ts_dir, 'human_classification_testset_worksheet_1.txt')
        sentinel = 'FROZEN WORKSHEET - DO NOT MODIFY'
        with open(ws_path, 'w') as f:
            f.write(sentinel)
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        with open(ws_path) as f:
            self.assertEqual(f.read(), sentinel)

    def test_idempotent(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        self._touch('01_transcripts', 'coded', 'coded_transcript_c1s1.txt')
        self._touch('04_validation', 'human_classification_c1s1.txt')
        self._touch('04_validation', 'content_validity_test_set.jsonl')
        self._touch('04_validation', 'testsets', 'worksheet1_base.txt')
        r1 = legacy_migration.migrate_v25_to_v3(self.tmpdir)
        r2 = legacy_migration.migrate_v25_to_v3(self.tmpdir)
        # Second run should move/delete nothing
        self.assertEqual(r2.get('diarized_moved', 0), 0)
        self.assertEqual(r2.get('coded_moved', 0), 0)
        self.assertEqual(r2.get('human_classification_moved', 0), 0)
        self.assertEqual(r2.get('cv_files_moved', 0), 0)
        self.assertEqual(r2.get('legacy_base_removed', 0), 0)

    def test_skips_destination_if_already_exists(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        dst = os.path.join(self.tmpdir, '01_transcripts_inputs', 'c1s1.vtt')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w') as f:
            f.write('existing content')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        with open(dst) as f:
            self.assertEqual(f.read(), 'existing content')


class TestJSONLToSQLiteMigration(unittest.TestCase):
    """Full v3-JSONL project -> qra.db importer (migrate_jsonl_to_sqlite).

    Builds a complete legacy JSONL project (frozen segments + every overlay +
    manifest + testset metadata + content-validity testset), migrates it, and
    asserts every table/field round-trips — including JSON columns, bool columns,
    and the gnn abstain 3-state (NULL | 0 | 1) — and that originals are relocated.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._build_v3_jsonl_project(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seg_record(self, sid, session, idx, speaker):
        return {
            'segment_id': sid, 'trial_id': 'trial_A', 'participant_id': 'p1',
            'session_id': session, 'session_number': 1, 'cohort_id': 1,
            'session_variant': '', 'segment_index': idx,
            'start_time_ms': idx * 1000, 'end_time_ms': idx * 1000 + 900,
            'total_segments_in_session': 3, 'speaker': speaker,
            'text': f'text {sid}', 'word_count': 2,
            'speakers_in_segment': None, 'session_file': 'c1s1.json',
        }

    def _build_v3_jsonl_project(self, run_dir):
        from process import classifications_io as cio

        # 1. frozen segments (2 sessions; s2 is a therapist segment)
        for session, members in (('c1s1', [('s1', 'participant'), ('s2', 'therapist')]),
                                 ('c1s2', [('s3', 'participant')])):
            seg_dir = _paths.segmented_session_dir(run_dir, session)
            os.makedirs(seg_dir, exist_ok=True)
            with open(os.path.join(seg_dir, 'segments.jsonl'), 'w') as f:
                for i, (sid, spk) in enumerate(members):
                    f.write(json.dumps(self._seg_record(sid, session, i, spk)) + '\n')
            with open(os.path.join(seg_dir, 'segmentation_meta.json'), 'w') as f:
                json.dump({'params_hash': 'PH', 'ingest_timestamp': 'T0'}, f)

        # 2. overlays (theme / purer / codebook / gnn) — written to the exact
        #    legacy filenames the importer reads.
        cls_dir = _paths.classifications_dir(run_dir)
        os.makedirs(cls_dir, exist_ok=True)

        def _wjsonl(key, records):
            with open(os.path.join(cls_dir, cio.OVERLAY_FILENAMES[key]), 'w') as f:
                for r in records:
                    f.write(json.dumps(r) + '\n')

        _wjsonl('theme', [
            {'segment_id': 's1', 'primary_stage': 2, 'secondary_stage': None,
             'llm_confidence_primary': 0.8, 'llm_confidence_secondary': None,
             'llm_justification': 'because', 'rater_ids': ['m1', 'm2'],
             'rater_votes': [2, 2], 'agreement_level': 'unanimous',
             'agreement_fraction': 1.0, 'needs_review': False,
             'consensus_vote': 2, 'tie_broken_by_confidence': False,
             'llm_run_consistency': 1, 'secondary_agreement_level': None,
             'secondary_agreement_fraction': None},
            {'segment_id': 's3', 'primary_stage': 0, 'secondary_stage': 1,
             'llm_confidence_primary': 0.4, 'llm_confidence_secondary': 0.3,
             'llm_justification': '', 'rater_ids': ['m1', 'm2', 'm3'],
             'rater_votes': [0, 1, 0], 'agreement_level': 'majority',
             'agreement_fraction': 0.67, 'needs_review': True,
             'consensus_vote': 'ABSTAIN', 'tie_broken_by_confidence': True,
             'llm_run_consistency': 0, 'secondary_agreement_level': 'split',
             'secondary_agreement_fraction': 0.33},
        ])
        _wjsonl('purer', [
            {'segment_id': 's2', 'purer_primary': 1, 'purer_secondary': None,
             'purer_confidence_primary': 0.7, 'purer_confidence_secondary': None,
             'purer_justification': 'util', 'purer_run_consistency': 1,
             'purer_agreement_level': 'unanimous', 'purer_agreement_fraction': 1.0,
             'purer_needs_review': False, 'purer_rater_ids': ['m1'],
             'purer_rater_votes': [1]},
        ])
        _wjsonl('codebook', [
            {'segment_id': 's1',
             'codebook_labels_embedding': ['A1', 'B2'], 'codebook_labels_llm': ['A1'],
             'codebook_labels_ensemble': ['A1', 'B2'], 'codebook_disagreements': ['B2'],
             'codebook_confidence': {'A1': 0.9, 'B2': 0.5}},
        ])
        # gnn — exercise the abstain 3-state explicitly: False(0), True(1), None(NULL)
        _wjsonl('gnn', [
            {'segment_id': 's1', 'gnn_vaamr_pred': 2, 'gnn_vaamr_conf': 0.9,
             'gnn_vaamr_abstain': False, 'gnn_purer_pred': None,
             'gnn_purer_conf': None, 'gnn_purer_abstain': None,
             'gnn_label_source': 'gnn_trained'},
            {'segment_id': 's3', 'gnn_vaamr_pred': 0, 'gnn_vaamr_conf': 0.3,
             'gnn_vaamr_abstain': True, 'gnn_purer_pred': None,
             'gnn_purer_conf': None, 'gnn_purer_abstain': None,
             'gnn_label_source': 'gnn_scale_mode'},
            {'segment_id': 's2', 'gnn_vaamr_pred': None, 'gnn_vaamr_conf': None,
             'gnn_vaamr_abstain': None, 'gnn_purer_pred': 1,
             'gnn_purer_conf': 0.8, 'gnn_purer_abstain': False,
             'gnn_label_source': 'gnn_trained'},
        ])

        # 3. classification manifest
        with open(_paths.classification_manifest_path(run_dir), 'w') as f:
            json.dump({
                'theme': {'model': 'm', 'framework': {'name': 'vaamr', 'version': '1'},
                          'n_segments': 2},
                'purer': {'model': 'm', 'n_segments': 1},
            }, f)

        # 4. validation testset worksheet metadata
        ts_meta_dir = os.path.join(_paths.meta_dir(run_dir), 'testset_meta')
        os.makedirs(ts_meta_dir, exist_ok=True)
        _ts_meta_name = 'human_classification_testset_worksheet_1.meta.json'
        with open(os.path.join(ts_meta_dir, _ts_meta_name), 'w') as f:
            json.dump({'kind': 'vaamr', 'legacy_import': False, 'segments': [
                {'session_id': 'c1s1', 'seg_num': 1, 'sha256': 'abc'},
                {'session_id': 'c1s2', 'seg_num': 1, 'sha256': 'def'},
            ]}, f)

        # 5. content-validity testset
        cv_dir = os.path.join(_paths.content_validity_dir(run_dir), 'cv_vaamr_v1')
        os.makedirs(cv_dir, exist_ok=True)
        with open(os.path.join(cv_dir, 'manifest.json'), 'w') as f:
            json.dump({'name': 'cv_vaamr_v1', 'kind': 'vaamr',
                       'framework': {'name': 'vaamr', 'version': '1'},
                       'created_at': '2026-01-01'}, f)
        with open(os.path.join(cv_dir, 'items.jsonl'), 'w') as f:
            for i in range(2):
                f.write(json.dumps({'id': f'item{i}', 'text': f't{i}',
                                    'expected_stage': i, 'difficulty': 'clear',
                                    'source_field': 'exemplar',
                                    'content_sha256': f'sha{i}'}) + '\n')

    def test_segments_imported(self):
        from process import segments_io as sio
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        self.assertEqual(set(sio.list_segmented_sessions(self.tmpdir)), {'c1s1', 'c1s2'})
        segs = {s.segment_id: s for s in sio.read_session_segments(self.tmpdir, 'c1s1')}
        self.assertEqual(set(segs), {'s1', 's2'})
        self.assertEqual(segs['s2'].speaker, 'therapist')

    def test_counts_returned(self):
        counts = legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        self.assertEqual(counts['sessions'], 2)
        self.assertEqual(counts['segments'], 3)
        self.assertEqual(counts['overlays'].get('theme'), 2)
        self.assertEqual(counts['overlays'].get('purer'), 1)
        self.assertEqual(counts['overlays'].get('codebook'), 1)
        self.assertEqual(counts['overlays'].get('gnn'), 3)
        self.assertEqual(counts['manifest_keys'], 2)
        self.assertEqual(counts['testset_worksheets'], 1)
        self.assertEqual(counts['cv_testsets'], 1)

    def test_theme_overlay_roundtrip(self):
        from process import classifications_io as cio
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        recs = {r['segment_id']: r for r in cio.read_overlay(self.tmpdir, 'theme')}
        self.assertEqual(recs['s1']['rater_votes'], [2, 2])          # JSON list
        self.assertIs(recs['s1']['needs_review'], False)             # bool 0
        self.assertIs(recs['s3']['needs_review'], True)              # bool 1
        self.assertIs(recs['s3']['tie_broken_by_confidence'], True)
        self.assertEqual(recs['s3']['consensus_vote'], 'ABSTAIN')    # heterogeneous JSON

    def test_gnn_abstain_3state_preserved(self):
        from process import classifications_io as cio
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        recs = {r['segment_id']: r for r in cio.read_overlay(self.tmpdir, 'gnn')}
        self.assertIs(recs['s1']['gnn_vaamr_abstain'], False)   # stored 0
        self.assertIs(recs['s3']['gnn_vaamr_abstain'], True)    # stored 1
        self.assertIsNone(recs['s2']['gnn_vaamr_abstain'])      # stored NULL
        self.assertIsNone(recs['s1']['gnn_purer_abstain'])
        self.assertIs(recs['s2']['gnn_purer_abstain'], False)

    def test_codebook_json_roundtrip(self):
        from process import classifications_io as cio
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        rec = cio.read_overlay(self.tmpdir, 'codebook')[0]
        self.assertEqual(rec['codebook_labels_ensemble'], ['A1', 'B2'])
        self.assertEqual(rec['codebook_confidence'], {'A1': 0.9, 'B2': 0.5})

    def test_manifest_roundtrip(self):
        from process import classifications_io as cio
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        man = cio.read_classification_manifest(self.tmpdir)
        self.assertEqual(set(man), {'theme', 'purer'})
        self.assertEqual(man['theme']['framework']['name'], 'vaamr')

    def test_testset_and_cv_roundtrip(self):
        from process.assembly import content_validity as cv
        from process import db
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        with db.open_db(self.tmpdir) as conn:
            ws = conn.execute(
                "SELECT kind, n_items FROM testset_worksheets WHERE worksheet_n=1"
            ).fetchone()
            self.assertEqual(ws['kind'], 'vaamr')
            self.assertEqual(ws['n_items'], 2)
            shas = conn.execute(
                "SELECT sha256 FROM testset_items WHERE worksheet_n=1 ORDER BY item_num"
            ).fetchall()
            self.assertEqual([r['sha256'] for r in shas], ['abc', 'def'])
        man = cv.read_cv_manifest(self.tmpdir, 'cv_vaamr_v1')
        self.assertEqual(man['item_ids'], ['item0', 'item1'])
        self.assertEqual(man['content_sha256']['item1'], 'sha1')

    def test_preview_counts_matches_migration(self):
        preview = legacy_migration.preview_counts(self.tmpdir)
        self.assertEqual(preview['sessions'], 2)
        self.assertEqual(preview['segments'], 3)
        self.assertEqual(preview['overlays'].get('gnn'), 3)
        self.assertEqual(preview['testset_worksheets'], 1)
        self.assertEqual(preview['cv_testsets'], 1)

    def test_originals_relocated_and_idempotent(self):
        from process import db
        legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        moved = os.path.join(self.tmpdir, '_legacy_files', '01_transcripts',
                             'segmented', 'c1s1', 'segments.jsonl')
        self.assertTrue(os.path.isfile(moved))
        self.assertTrue(db.db_exists(self.tmpdir))
        # qra.db now exists, so re-running migrates nothing.
        counts2 = legacy_migration.migrate_jsonl_to_sqlite(self.tmpdir)
        self.assertEqual(counts2['segments'], 0)


if __name__ == '__main__':
    unittest.main()
