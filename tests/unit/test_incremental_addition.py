"""
Tests for the incremental data-addition feature in QRA.

Covers:
  - merge_overlay / per-key wrappers in process/classifications_io.py
  - peek_speaker_labels in process/transcript_ingestion.py
  - _next_id in process/speaker_walkthrough.py
  - resolve_pinned_classifier_config in process/orchestrator.py
  - stage_classify_theme with only_session_ids (manifest markers)
  - Byte-identical preservation of unmodified overlay rows after incremental run
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


def _make_segment(segment_id='seg_001', session_id='c1s1', speaker='participant', **kwargs):
    defaults = dict(
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
        total_segments_in_session=3,
        speaker=speaker,
        text='Test segment text.',
        word_count=3,
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


# ---------------------------------------------------------------------------
# TestClass 1: TestMergeOverlay
# ---------------------------------------------------------------------------

class TestMergeOverlay(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _classified_theme_seg(self, seg_id, primary, session_id='c1s1'):
        seg = _make_segment(segment_id=seg_id, session_id=session_id)
        seg.primary_stage = primary
        seg.secondary_stage = None
        seg.llm_confidence_primary = 0.9
        seg.llm_confidence_secondary = None
        seg.llm_justification = 'Test justification.'
        seg.rater_ids = ['rater_1']
        seg.rater_votes = [{'rater': 'rater_1', 'stage': primary, 'confidence': 0.9}]
        seg.agreement_level = 'unanimous'
        seg.agreement_fraction = 1.0
        seg.needs_review = False
        seg.consensus_vote = primary
        seg.tie_broken_by_confidence = False
        seg.llm_run_consistency = 2
        return seg

    def _classified_purer_seg(self, seg_id, primary, session_id='c1s1'):
        seg = _make_segment(segment_id=seg_id, session_id=session_id, speaker='therapist')
        seg.purer_primary = primary
        seg.purer_secondary = None
        seg.purer_confidence_primary = 0.85
        seg.purer_confidence_secondary = None
        seg.purer_justification = 'PURER justification.'
        seg.purer_run_consistency = 2
        seg.purer_agreement_level = 'unanimous'
        seg.purer_agreement_fraction = 1.0
        seg.purer_needs_review = False
        seg.purer_rater_ids = ['rater_1']
        seg.purer_rater_votes = [{'rater': 'rater_1', 'stage': primary}]
        return seg

    def _classified_codebook_seg(self, seg_id, session_id='c1s1'):
        seg = _make_segment(segment_id=seg_id, session_id=session_id)
        seg.codebook_labels_embedding = ['VE.1']
        seg.codebook_labels_llm = ['VE.1']
        seg.codebook_labels_ensemble = ['VE.1']
        seg.codebook_disagreements = []
        seg.codebook_confidence = {'VE.1': 0.9}
        return seg

    def _read_jsonl(self, path):
        with open(path, encoding='utf-8') as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    # --- theme overlay ---

    def test_merge_into_empty_overlay(self):
        """merge_theme_overlay on a fresh dir creates the file with sorted rows."""
        from process import classifications_io as cio
        seg = self._classified_theme_seg('seg_001', primary=2)
        path = cio.merge_theme_overlay(self.tmpdir, [seg])
        self.assertTrue(os.path.isfile(path))
        rows = self._read_jsonl(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['segment_id'], 'seg_001')
        self.assertEqual(rows[0]['primary_stage'], 2)

    def test_merge_appends_new_segment_ids(self):
        """write initial overlay with [seg_001, seg_002], then merge [seg_003] — all three present sorted."""
        from process import classifications_io as cio
        seg1 = self._classified_theme_seg('seg_001', primary=1)
        seg2 = self._classified_theme_seg('seg_002', primary=2)
        cio.write_theme_overlay(self.tmpdir, [seg1, seg2])

        seg3 = self._classified_theme_seg('seg_003', primary=3)
        cio.merge_theme_overlay(self.tmpdir, [seg3])

        rows = self._read_jsonl(cio.overlay_path(self.tmpdir, 'theme'))
        ids = [r['segment_id'] for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertIn('seg_001', ids)
        self.assertIn('seg_002', ids)
        self.assertIn('seg_003', ids)
        self.assertEqual(len(rows), 3)

        # existing rows are preserved with original values
        by_id = {r['segment_id']: r for r in rows}
        self.assertEqual(by_id['seg_001']['primary_stage'], 1)
        self.assertEqual(by_id['seg_002']['primary_stage'], 2)
        self.assertEqual(by_id['seg_003']['primary_stage'], 3)

    def test_merge_replaces_matching_segment_ids(self):
        """write initial overlay with seg_001.primary_stage=1, then merge with primary_stage=4."""
        from process import classifications_io as cio
        seg = self._classified_theme_seg('seg_001', primary=1)
        cio.write_theme_overlay(self.tmpdir, [seg])

        updated_seg = self._classified_theme_seg('seg_001', primary=4)
        cio.merge_theme_overlay(self.tmpdir, [updated_seg])

        rows = self._read_jsonl(cio.overlay_path(self.tmpdir, 'theme'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['primary_stage'], 4)

    def test_merge_preserves_unrelated_rows(self):
        """write [seg_a, seg_b, seg_c], merge [seg_b changed] — seg_a and seg_c unchanged."""
        from process import classifications_io as cio
        seg_a = self._classified_theme_seg('seg_a', primary=0)
        seg_b = self._classified_theme_seg('seg_b', primary=1)
        seg_c = self._classified_theme_seg('seg_c', primary=2)
        cio.write_theme_overlay(self.tmpdir, [seg_a, seg_b, seg_c])

        seg_b_updated = self._classified_theme_seg('seg_b', primary=4)
        cio.merge_theme_overlay(self.tmpdir, [seg_b_updated])

        rows = self._read_jsonl(cio.overlay_path(self.tmpdir, 'theme'))
        self.assertEqual(len(rows), 3)
        by_id = {r['segment_id']: r for r in rows}
        self.assertEqual(by_id['seg_a']['primary_stage'], 0)
        self.assertEqual(by_id['seg_b']['primary_stage'], 4)
        self.assertEqual(by_id['seg_c']['primary_stage'], 2)

    def test_merge_purer_overlay_round_trip(self):
        """merge_purer_overlay: create, then merge a new segment — both appear sorted."""
        from process import classifications_io as cio
        seg1 = self._classified_purer_seg('th_001', primary=0)
        cio.write_purer_overlay(self.tmpdir, [seg1])

        seg2 = self._classified_purer_seg('th_002', primary=2)
        cio.merge_purer_overlay(self.tmpdir, [seg2])

        rows = self._read_jsonl(cio.overlay_path(self.tmpdir, 'purer'))
        ids = [r['segment_id'] for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(rows), 2)
        by_id = {r['segment_id']: r for r in rows}
        self.assertEqual(by_id['th_001']['purer_primary'], 0)
        self.assertEqual(by_id['th_002']['purer_primary'], 2)

    def test_merge_codebook_overlay_round_trip(self):
        """merge_codebook_overlay: create, then merge a new segment — both appear."""
        from process import classifications_io as cio
        seg1 = self._classified_codebook_seg('seg_001')
        cio.write_codebook_overlay(self.tmpdir, [seg1])

        seg2 = self._classified_codebook_seg('seg_002')
        seg2.codebook_labels_ensemble = ['VE.2']
        cio.merge_codebook_overlay(self.tmpdir, [seg2])

        rows = self._read_jsonl(cio.overlay_path(self.tmpdir, 'codebook'))
        ids = [r['segment_id'] for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(rows), 2)
        by_id = {r['segment_id']: r for r in rows}
        self.assertEqual(by_id['seg_001']['codebook_labels_ensemble'], ['VE.1'])
        self.assertEqual(by_id['seg_002']['codebook_labels_ensemble'], ['VE.2'])

    def test_merge_cv_overlay_round_trip(self):
        """merge_cross_validation_overlay: file is created and is valid JSONL."""
        from process import classifications_io as cio
        seg = _make_segment('seg_001')
        cio.write_cross_validation_overlay(self.tmpdir, [seg])

        seg2 = _make_segment('seg_002')
        cio.merge_cross_validation_overlay(self.tmpdir, [seg2])

        path = cio.overlay_path(self.tmpdir, 'cv')
        self.assertTrue(os.path.isfile(path))
        rows = self._read_jsonl(path)
        # File must be valid JSONL with 2 rows
        self.assertEqual(len(rows), 2)
        ids = [r['segment_id'] for r in rows]
        self.assertIn('seg_001', ids)
        self.assertIn('seg_002', ids)


# ---------------------------------------------------------------------------
# TestClass 2: TestPeekSpeakerLabels
# ---------------------------------------------------------------------------

class TestPeekSpeakerLabels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_diarized_json(self, filename, sentences):
        data = {
            'metadata': {'duration': 100},
            'sentences': sentences,
        }
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh)
        return path

    def test_peek_returns_speakers_with_samples(self):
        """Three speakers, each with multiple sentences; all keys returned with valid samples."""
        from process.transcript_ingestion import peek_speaker_labels
        sentences = []
        for i in range(6):
            sentences.append({'text': f'Alice says something quite meaningful about the topic here number {i}.', 'speaker': 'Alice', 'start': float(i * 5), 'end': float(i * 5 + 4)})
        for i in range(4):
            sentences.append({'text': f'Bob contributes his perspective on the overall discussion today number {i}.', 'speaker': 'Bob', 'start': float(30 + i * 5), 'end': float(30 + i * 5 + 4)})
        for i in range(3):
            sentences.append({'text': f'Carol adds her thoughts about mindfulness and attention regulation now {i}.', 'speaker': 'Carol', 'start': float(50 + i * 5), 'end': float(50 + i * 5 + 4)})

        path = self._write_diarized_json('session.json', sentences)
        result = peek_speaker_labels(path, max_samples_per_speaker=5, max_chars_per_sample=200)

        self.assertIn('Alice', result)
        self.assertIn('Bob', result)
        self.assertIn('Carol', result)

        for speaker, samples in result.items():
            self.assertIsInstance(samples, list)
            self.assertLessEqual(len(samples), 5, f"{speaker}: too many samples")
            for s in samples:
                self.assertLessEqual(len(s), 201, f"{speaker}: sample too long (allows for ellipsis)")

    def test_peek_samples_truncated_at_char_limit(self):
        """A single very-long utterance (>250 chars) should appear truncated with ellipsis."""
        from process.transcript_ingestion import peek_speaker_labels
        long_text = 'x' * 300
        sentences = [{'text': long_text, 'speaker': 'Alice', 'start': 0.0, 'end': 5.0}]
        path = self._write_diarized_json('long_session.json', sentences)
        result = peek_speaker_labels(path, max_samples_per_speaker=5, max_chars_per_sample=200)

        self.assertIn('Alice', result)
        self.assertTrue(len(result['Alice']) > 0)
        sample = result['Alice'][0]
        # The sample should be truncated at 200 chars + ellipsis char
        self.assertTrue(sample.endswith('…'), f"Expected truncation with '…', got: {sample!r}")
        # The base text portion should be exactly max_chars_per_sample
        self.assertEqual(len(sample), 201)  # 200 chars + 1-byte ellipsis (unicode)

    def test_peek_speaker_count_order(self):
        """Speaker with more utterances appears first (insertion order reflects desc count)."""
        from process.transcript_ingestion import peek_speaker_labels
        sentences = []
        # Bob gets 8 utterances, Alice gets 3
        for i in range(8):
            sentences.append({'text': f'Bob says something important about this topic here number {i}.', 'speaker': 'Bob', 'start': float(i * 5), 'end': float(i * 5 + 4)})
        for i in range(3):
            sentences.append({'text': f'Alice contributes her perspective on the discussion now {i}.', 'speaker': 'Alice', 'start': float(40 + i * 5), 'end': float(40 + i * 5 + 4)})

        path = self._write_diarized_json('order_session.json', sentences)
        result = peek_speaker_labels(path)

        speakers = list(result.keys())
        self.assertGreater(len(speakers), 1)
        self.assertEqual(speakers[0], 'Bob', f"Expected Bob first (more utterances), got: {speakers}")

    def test_peek_missing_file_returns_empty(self):
        """peek_speaker_labels on a nonexistent file returns {}."""
        from process.transcript_ingestion import peek_speaker_labels
        result = peek_speaker_labels('/nonexistent/file.json')
        self.assertEqual(result, {})

    def test_peek_vtt_format(self):
        """A minimal WebVTT file with 'Speaker: text' cues is parsed correctly."""
        from process.transcript_ingestion import peek_speaker_labels
        vtt_content = (
            "WEBVTT\n"
            "\n"
            "1\n"
            "00:00:00.000 --> 00:00:05.000\n"
            "Alice: Hello everyone today we are going to explore mindfulness techniques.\n"
            "\n"
            "2\n"
            "00:00:05.000 --> 00:00:10.000\n"
            "Bob: Yes I agree with that approach and would like to add some thoughts here.\n"
            "\n"
            "3\n"
            "00:00:10.000 --> 00:00:15.000\n"
            "Alice: Let us now focus on the breath and body sensations present right now.\n"
            "\n"
            "4\n"
            "00:00:15.000 --> 00:00:20.000\n"
            "Bob: That resonates with my experience of chronic pain and daily practice.\n"
        )
        vtt_path = os.path.join(self.tmpdir, 'session.vtt')
        with open(vtt_path, 'w', encoding='utf-8') as fh:
            fh.write(vtt_content)

        result = peek_speaker_labels(vtt_path)
        self.assertIn('Alice', result)
        self.assertIn('Bob', result)
        # Each speaker has 2 utterances, so at least 1 sample each
        self.assertGreater(len(result['Alice']), 0)
        self.assertGreater(len(result['Bob']), 0)


# ---------------------------------------------------------------------------
# TestClass 3: TestNextId
# ---------------------------------------------------------------------------

class TestNextId(unittest.TestCase):

    def test_next_participant_id_increments_max(self):
        """_next_id picks max existing participant number and increments by 1."""
        from process.speaker_walkthrough import _next_id
        existing = {
            'a': ('participant', 'Participant_MM005'),
            'b': ('participant', 'Participant_MM003'),
        }
        result = _next_id(existing, 'participant')
        self.assertEqual(result, 'Participant_MM006')

    def test_next_therapist_id_increments(self):
        """_next_id increments the highest therapist number."""
        from process.speaker_walkthrough import _next_id
        existing = {
            'alice': ('therapist', 'therapist_2'),
            'bob': ('therapist', 'therapist_5'),
        }
        result = _next_id(existing, 'therapist')
        self.assertEqual(result, 'therapist_6')

    def test_next_staff_id_starts_at_one(self):
        """Empty map → staff_1 for role 'staff'."""
        from process.speaker_walkthrough import _next_id
        result = _next_id({}, 'staff')
        self.assertEqual(result, 'staff_1')

    def test_next_id_ignores_other_roles_for_max(self):
        """A map containing only therapists → participant starts at Participant_MM001."""
        from process.speaker_walkthrough import _next_id
        existing = {
            'therapist_alice': ('therapist', 'therapist_10'),
        }
        result = _next_id(existing, 'participant')
        self.assertEqual(result, 'Participant_MM001')

    def test_next_id_handles_legacy_participant_format(self):
        """Map contains 'participant_7' (lowercase legacy) → next is Participant_MM008."""
        from process.speaker_walkthrough import _next_id
        existing = {
            'legacy_label': ('participant', 'participant_7'),
        }
        result = _next_id(existing, 'participant')
        self.assertEqual(result, 'Participant_MM008')


# ---------------------------------------------------------------------------
# TestClass 4: TestResolvePinnedClassifierConfig
# ---------------------------------------------------------------------------

class TestResolvePinnedClassifierConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_config(self, theme_model='default-model'):
        from process.config import PipelineConfig
        config = PipelineConfig(output_dir=self.tmpdir)
        config.theme_classification.model = theme_model
        return config

    def test_no_manifest_returns_same_config(self):
        """Empty dir (no manifest) → resolve returns the exact same config object."""
        from process.orchestrator import resolve_pinned_classifier_config
        config = self._make_config()
        result = resolve_pinned_classifier_config(self.tmpdir, 'theme', config)
        self.assertIs(result, config)

    def test_no_manifest_entry_for_key_returns_same(self):
        """Manifest exists with only 'purer' key → asking for 'theme' returns same config."""
        from process import classifications_io as cio
        from process.orchestrator import resolve_pinned_classifier_config
        cio.update_classification_manifest(self.tmpdir, key='purer', entry={'model': 'purer-model', 'n_segments': 1})

        config = self._make_config('my-theme-model')
        result = resolve_pinned_classifier_config(self.tmpdir, 'theme', config)
        self.assertIs(result, config)

    def test_pinning_overrides_model(self):
        """Manifest entry with model → returned config has that model; original not mutated."""
        from process import classifications_io as cio
        from process.orchestrator import resolve_pinned_classifier_config
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'gpt-pinned', 'n_segments': 5}
        )

        config = self._make_config('different-model')
        result = resolve_pinned_classifier_config(self.tmpdir, 'theme', config)

        # Returned config has pinned model
        self.assertEqual(result.theme_classification.model, 'gpt-pinned')
        # Original config is NOT mutated
        self.assertEqual(config.theme_classification.model, 'different-model')
        # Returned config is a different object
        self.assertIsNot(result, config)

    def test_pinning_no_change_when_models_match(self):
        """When current model already matches manifest, result has the same model value."""
        from process import classifications_io as cio
        from process.orchestrator import resolve_pinned_classifier_config
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'same-model', 'n_segments': 3}
        )

        config = self._make_config('same-model')
        result = resolve_pinned_classifier_config(self.tmpdir, 'theme', config)

        # Either same object or deep copy — either way, model value is unchanged
        self.assertEqual(result.theme_classification.model, 'same-model')

    def test_cv_key_returns_same_config(self):
        """'cv' has no per-classifier sub-attr → returns current_config unchanged."""
        from process import classifications_io as cio
        from process.orchestrator import resolve_pinned_classifier_config
        cio.update_classification_manifest(
            self.tmpdir, key='cv', entry={'model': 'some-model', 'n_segments': 2}
        )

        config = self._make_config()
        result = resolve_pinned_classifier_config(self.tmpdir, 'cv', config)
        # cv key has no sub_attr → returns current_config as-is
        self.assertIs(result, config)


# ---------------------------------------------------------------------------
# TestClass 5: TestIncrementalManifestMarker
# ---------------------------------------------------------------------------

class TestIncrementalManifestMarker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_subset_classification_adds_incremental_markers(self):
        """stage_classify_theme with only_session_ids adds last_incremental_at + n_new_segments."""
        from process import classifications_io as cio
        from process.orchestrator import stage_classify_theme

        seg_a = _make_segment(segment_id='seg_a', session_id='c1s1')
        seg_a.primary_stage = 1
        seg_b = _make_segment(segment_id='seg_b', session_id='c1s2')
        seg_b.primary_stage = 2

        # First populate the overlay with both segments to simulate "old" state
        cio.write_theme_overlay(self.tmpdir, [seg_a, seg_b])
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'baseline', 'n_segments': 2}
        )

        # Now do an incremental subset — segment 'seg_c' is in session c1s3
        seg_c = _make_segment(segment_id='seg_c', session_id='c1s3')
        seg_c.primary_stage = 3
        stage_classify_theme(
            None, None,
            segments=[seg_a, seg_b, seg_c],
            output_dir=self.tmpdir,
            only_session_ids={'c1s3'},
        )

        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIsNotNone(m)
        self.assertIn('theme', m)
        self.assertIn('last_incremental_at', m['theme'])
        self.assertEqual(m['theme']['n_new_segments'], 1)

    def test_full_run_does_not_add_incremental_markers(self):
        """stage_classify_theme without only_session_ids does NOT add incremental markers."""
        from process import classifications_io as cio
        from process.orchestrator import stage_classify_theme

        seg_a = _make_segment(segment_id='seg_a', session_id='c1s1')
        seg_a.primary_stage = 1

        stage_classify_theme(
            None, None,
            segments=[seg_a],
            output_dir=self.tmpdir,
        )

        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIsNotNone(m)
        self.assertNotIn('last_incremental_at', m.get('theme', {}))
        self.assertNotIn('n_new_segments', m.get('theme', {}))


# ---------------------------------------------------------------------------
# TestClass 6: TestSubsetOverlayMerge
# ---------------------------------------------------------------------------

class TestSubsetOverlayMerge(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_subset_merge_preserves_old_rows_byte_identical(self):
        """After incremental run, overlay rows for old sessions preserve their original values."""
        from process import classifications_io as cio
        from process.orchestrator import stage_classify_theme

        seg_a = _make_segment(segment_id='seg_a', session_id='c1s1')
        seg_a.primary_stage = 1
        seg_b = _make_segment(segment_id='seg_b', session_id='c1s2')
        seg_b.primary_stage = 2

        # Populate initial overlay
        cio.write_theme_overlay(self.tmpdir, [seg_a, seg_b])
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'baseline', 'n_segments': 2}
        )

        # Read the initial overlay rows for seg_a and seg_b
        with open(cio.overlay_path(self.tmpdir, 'theme'), encoding='utf-8') as fh:
            initial_rows = {json.loads(ln)['segment_id']: json.loads(ln)
                            for ln in fh if ln.strip()}

        # Run incremental classification for a new session c1s3
        seg_c = _make_segment(segment_id='seg_c', session_id='c1s3')
        seg_c.primary_stage = 3
        stage_classify_theme(
            None, None,
            segments=[seg_a, seg_b, seg_c],
            output_dir=self.tmpdir,
            only_session_ids={'c1s3'},
        )

        # Read the updated overlay
        with open(cio.overlay_path(self.tmpdir, 'theme'), encoding='utf-8') as fh:
            updated_rows = {json.loads(ln)['segment_id']: json.loads(ln)
                            for ln in fh if ln.strip()}

        # All three segments present
        self.assertIn('seg_a', updated_rows)
        self.assertIn('seg_b', updated_rows)
        self.assertIn('seg_c', updated_rows)

        # Old rows are byte-identical (same JSON values for all fields)
        self.assertEqual(updated_rows['seg_a'], initial_rows['seg_a'],
                         "seg_a row should be byte-identical after incremental run")
        self.assertEqual(updated_rows['seg_b'], initial_rows['seg_b'],
                         "seg_b row should be byte-identical after incremental run")

        # New row has the correct primary_stage
        self.assertEqual(updated_rows['seg_c']['primary_stage'], 3)

    def test_incremental_does_not_overwrite_full_session_data(self):
        """Only the targeted session's segments are touched; others remain with original stage."""
        from process import classifications_io as cio
        from process.orchestrator import stage_classify_theme

        seg_x = _make_segment(segment_id='seg_x', session_id='c2s1')
        seg_x.primary_stage = 0
        seg_y = _make_segment(segment_id='seg_y', session_id='c2s1')
        seg_y.primary_stage = 0

        cio.write_theme_overlay(self.tmpdir, [seg_x, seg_y])
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'baseline', 'n_segments': 2}
        )

        # Incremental run adds a segment from a completely different session
        seg_new = _make_segment(segment_id='seg_new', session_id='c2s2')
        seg_new.primary_stage = 4

        stage_classify_theme(
            None, None,
            segments=[seg_x, seg_y, seg_new],
            output_dir=self.tmpdir,
            only_session_ids={'c2s2'},
        )

        # Read result
        with open(cio.overlay_path(self.tmpdir, 'theme'), encoding='utf-8') as fh:
            rows = {json.loads(ln)['segment_id']: json.loads(ln) for ln in fh if ln.strip()}

        self.assertEqual(rows['seg_x']['primary_stage'], 0, "seg_x must be untouched")
        self.assertEqual(rows['seg_y']['primary_stage'], 0, "seg_y must be untouched")
        self.assertEqual(rows['seg_new']['primary_stage'], 4, "seg_new must have new stage")


if __name__ == '__main__':
    unittest.main()
