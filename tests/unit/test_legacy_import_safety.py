"""
test_legacy_import_safety.py
-----------------------------
Comprehensive legacy-import safety test suite.

VALIDATES: Updating a project created on the `working` git branch using
the new modular `beta` branch without:
  1. Changing segment text or segment numbering
  2. Altering human-validated testset content
  3. Re-segmenting transcripts
  4. Damaging frozen segmentation artifacts
  5. Breaking legacy config parsing
  6. Corrupting content-SHA locking

ARCHITECTURE — what the beta codebase does when it sees a legacy project:
  a) legacy_migration.is_legacy_project() → detects pre-modular layout
  b) legacy_migration.migrate_legacy_segments() → extracts per-session frozen segments
     from master_segments.jsonl with params_hash='legacy-pre-modular'
  c) legacy_migration.is_v25_layout() + migrate_v25_to_v3() → moves old-path
     artifacts to v3 locations (full_transcripts/, content_validity/, inputs/)
  d) stage_ingest() → recognizes 'legacy-pre-modular' hash, skips re-segmentation
  e) classify stages → read frozen segments, write overlays, never touch frozen
  f) assemble → reads frozen + overlays, builds master_segments
  g) validate → refreshes AI answer keys WITH content-sha verification

SAFETY INVARIANTS tested by this suite:
  S1: segment_id, segment_index, session_id, text NEVER change on import
  S2: segment text SHA-256 is captured at migration and verified on every refresh
  S3: human_worksheet content is immutable (text, segment IDs, item ordering)
  S4: AI_classification_worksheet can be updated with new classifications
  S5: legacy config parses correctly without losing any data
  S6: frozen guard prevents silent overwrite of segments or testsets
  S7: create_missing=False prevents new testset creation during assemble
  S8: overlay I/O never contaminates segment identity fields
  S9: master_segments compatibility across working→beta
"""

import copy
import datetime
import hashlib
import json
import os
import re
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field, asdict


# ============================================================================
# Fixture helpers
# ============================================================================

@dataclass
class MockSegment:
    """Minimal Segment mock matching the real dataclass field set."""
    segment_id: str = ""
    trial_id: str = "MMORE"
    participant_id: str = ""
    session_id: str = ""
    session_number: int = 0
    cohort_id: Optional[int] = None
    session_variant: str = ""
    segment_index: int = 0
    start_time_ms: int = 0
    end_time_ms: int = 0
    total_segments_in_session: int = 0
    speaker: str = ""
    text: str = ""
    word_count: int = 0
    speakers_in_segment: Optional[List[str]] = None
    session_file: str = ""

    # VAAMR classification fields
    primary_stage: Optional[int] = None
    secondary_stage: Optional[int] = None
    llm_confidence_primary: Optional[float] = None
    llm_confidence_secondary: Optional[float] = None
    llm_justification: Optional[str] = None
    rater_ids: Optional[List[str]] = None
    rater_votes: Optional[List[dict]] = None
    agreement_level: Optional[str] = None
    agreement_fraction: Optional[float] = None
    needs_review: bool = False
    consensus_vote: Optional[str] = None
    tie_broken_by_confidence: bool = False
    llm_run_consistency: Optional[int] = None
    secondary_agreement_level: Optional[str] = None
    secondary_agreement_fraction: Optional[float] = None

    # PURER classification fields
    purer_primary: Optional[int] = None
    purer_secondary: Optional[int] = None
    purer_confidence_primary: Optional[float] = None
    purer_confidence_secondary: Optional[float] = None
    purer_justification: Optional[str] = None
    purer_run_consistency: Optional[int] = None
    purer_agreement_level: Optional[str] = None
    purer_agreement_fraction: Optional[float] = None
    purer_needs_review: bool = False
    purer_rater_ids: Optional[List[str]] = None
    purer_rater_votes: Optional[List[dict]] = None

    # Codebook fields
    codebook_labels_embedding: Optional[List] = None
    codebook_labels_llm: Optional[List] = None
    codebook_labels_ensemble: Optional[List] = None
    codebook_disagreements: Optional[List] = None
    codebook_confidence: Optional[float] = None

    # Validation fields
    human_label: Optional[int] = None
    human_secondary_label: Optional[int] = None
    adjudicated_label: Optional[int] = None
    in_human_coded_subset: bool = False
    label_status: str = "llm_only"
    final_label: Optional[int] = None
    final_label_source: Optional[str] = None
    label_confidence_tier: Optional[str] = None


def _make_segment(
    segment_id="c1s1_001",
    session_id="c1s1",
    segment_index=0,
    text="Sample text for testing",
    word_count=5,
    participant_id="Participant_MM001",
    speaker="Participant_MM001",
    start_time_ms=0,
    end_time_ms=60000,
    trial_id="MMORE",
    session_number=1,
    cohort_id=1,
    **kwargs,
) -> MockSegment:
    seg = MockSegment(
        segment_id=segment_id, session_id=session_id, segment_index=segment_index,
        text=text, word_count=word_count, participant_id=participant_id,
        speaker=speaker, start_time_ms=start_time_ms, end_time_ms=end_time_ms,
        trial_id=trial_id, session_number=session_number, cohort_id=cohort_id,
    )
    for k, v in kwargs.items():
        setattr(seg, k, v)
    return seg


# ============================================================================
# Config roundtrip tests
# ============================================================================

class TestLegacyConfigRoundtrip(unittest.TestCase):
    """Verify the beta-branch PipelineConfig.from_json() correctly loads
    every field from a working-branch qra_config.json."""

    @classmethod
    def legacy_config_dict(cls):
        """Exact replica of MMORE_Processed/02_meta/qra_config.json (working branch)."""
        return {
            "pipeline": {
                "transcript_dir": "./data/MMORE/",
                "output_dir": "./data/MMORE_Processed/",
                "trial_id": "MoveMORE",
                "speaker_anonymization_key_path": "/home/wisgood/qra/Qualitative_Research_Algorithm/data/full/speaker_anonymization_key.json",
                "run_codebook_classifier": False,
                "auto_analyze": True,
            },
            "speaker_filter": {
                "mode": "exclude",
                "speakers": [
                    "Move-MORE Study", "Anand", "Lani",
                    "Wade (Study Coordinator)", "Rebecca Heron", "Michelle Berg",
                ],
            },
            "run_purer_labeler": True,
            "purer_cue": {"skip_lesson_content": False},
            "theme_classification": {
                "backend": "lmstudio",
                "model": "qwen/qwen3-next-80b",
                "summarization_model": "nvidia/nemotron-3-nano-4b",
                "lmstudio_base_url": "http://10.0.0.58:1234/v1",
                "n_runs": 3,
                "temperature": 0.1,
                "per_run_models": [
                    "qwen/qwen3-next-80b",
                    "google/gemma-4-31b",
                    "nvidia/nemotron-3-super",
                ],
            },
            "purer_classification": {
                "backend": "lmstudio",
                "model": "google/gemma-4-31b",
                "summarization_model": "nvidia/nemotron-3-nano-4b",
                "lmstudio_base_url": "http://10.0.0.58:1234/v1",
                "n_runs": 1,
                "temperature": 0.1,
                "per_run_models": [],
            },
            "segmentation": {
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "max_gap_seconds": 15.0,
                "min_words_per_sentence": 20,
                "max_segment_duration_seconds": 60.0,
                "min_segment_words_conversational": 60,
                "max_segment_words_conversational": 500,
                "use_adaptive_threshold": True,
                "min_prominence": 0.05,
                "use_topic_clustering": True,
                "use_llm_refinement": True,
                "llm_refinement_mode": "full",
            },
            "framework": {"preset": "vaamr"},
            "confidence_tiers": {
                "high_confidence": 0.8,
                "medium_min_confidence": 0.6,
            },
            "test_sets": {
                "enabled": True,
                "n_sets": 2,
                "fraction_per_set": 0.1,
                "random_seed": 42,
            },
            "therapist_cues": {
                "enabled": True,
                "max_length_per_cue": 250,
                "max_length_of_average_cue_responses": 300,
            },
            "session_summaries": {"enabled": True, "max_words_per_session": 300},
            "participant_summaries": {"enabled": True, "max_words_per_session": 300},
        }

    def _load_cfg(self):
        from process.config import PipelineConfig
        from qra import _flatten_wizard_config
        flat = _flatten_wizard_config(self.legacy_config_dict())
        return PipelineConfig.from_json(flat)

    def test_load_legacy_config_no_exception(self):
        """Legacy config must parse without error."""
        try:
            self._load_cfg()
        except Exception as e:
            self.fail(f"Legacy config failed to parse: {e}")

    def test_flat_test_sets_to_test_sets_config(self):
        """Flat {enabled, n_sets, fraction_per_set, random_seed} → TestSetsConfig
        with vaamr enabled.  This is the critical back-compat path."""
        from process.config import TestSetsConfig, TestSetSpec, _parse_test_sets_config
        from qra import _flatten_wizard_config
        raw = self.legacy_config_dict()
        flat = _flatten_wizard_config(raw)
        self.assertIn('test_sets', flat)
        self.assertIsInstance(flat['test_sets'], dict)
        self.assertIn('enabled', flat['test_sets'])
        self.assertNotIn('vaamr', flat['test_sets'])
        ts_cfg = _parse_test_sets_config(flat['test_sets'])
        self.assertIsInstance(ts_cfg, TestSetsConfig)
        self.assertTrue(ts_cfg.vaamr.enabled)
        self.assertEqual(ts_cfg.vaamr.n_sets, 2)
        self.assertEqual(ts_cfg.vaamr.fraction_per_set, 0.1)
        self.assertEqual(ts_cfg.vaamr.random_seed, 42)
        self.assertEqual(ts_cfg.vaamr.name, 'vaamr_testset')
        self.assertFalse(ts_cfg.purer.enabled)
        self.assertFalse(ts_cfg.codebook.enabled)

    def test_legacy_trial_id_preserved(self):
        cfg = self._load_cfg()
        self.assertEqual(cfg.trial_id, "MoveMORE")

    def test_legacy_segmentation_params_preserved(self):
        cfg = self._load_cfg()
        seg = cfg.segmentation
        self.assertEqual(seg.embedding_model, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(seg.max_gap_seconds, 15.0)
        self.assertEqual(seg.min_words_per_sentence, 20)
        self.assertEqual(seg.max_segment_duration_seconds, 60.0)
        self.assertEqual(seg.min_segment_words_conversational, 60)
        self.assertEqual(seg.max_segment_words_conversational, 500)
        self.assertTrue(seg.use_adaptive_threshold)
        self.assertEqual(seg.min_prominence, 0.05)
        self.assertTrue(seg.use_topic_clustering)
        self.assertTrue(seg.use_llm_refinement)
        self.assertEqual(seg.llm_refinement_mode, "full")

    def test_legacy_speaker_anonymization_path_preserved(self):
        cfg = self._load_cfg()
        self.assertEqual(
            cfg.speaker_anonymization_key_path,
            "/home/wisgood/qra/Qualitative_Research_Algorithm/data/full/speaker_anonymization_key.json",
        )

    def test_legacy_auto_analyze_preserved(self):
        cfg = self._load_cfg()
        self.assertTrue(cfg.auto_analyze)

    def test_legacy_speaker_filter_preserved(self):
        cfg = self._load_cfg()
        self.assertEqual(cfg.speaker_filter.mode, "exclude")
        self.assertIn("Move-MORE Study", cfg.speaker_filter.speakers)
        self.assertIn("Wade (Study Coordinator)", cfg.speaker_filter.speakers)
        self.assertEqual(len(cfg.speaker_filter.speakers), 6)

    def test_missing_content_validity_defaults(self):
        from qra import _flatten_wizard_config
        raw = self.legacy_config_dict()
        flat = _flatten_wizard_config(raw)
        self.assertNotIn('content_validity', flat)
        cfg = self._load_cfg()
        self.assertTrue(cfg.content_validity.vaamr.enabled)
        self.assertEqual(cfg.content_validity.vaamr.name, 'cv_vaamr_v1')
        self.assertFalse(cfg.content_validity.purer.enabled)

    def test_legacy_conf_to_json_no_secrets(self):
        cfg = self._load_cfg()
        data = cfg.to_json()
        flat_str = json.dumps(data)
        # api_key field must be present but blanked (empty string), never a real secret
        self.assertNotIn('"api_key": "sk-', flat_str)
        import re as _re
        for m in _re.finditer(r'"api_key"\s*:\s*"([^"]*)"', flat_str):
            self.assertEqual(m.group(1), '', f"api_key should be blank, got: {m.group(1)!r}")

    def test_legacy_conf_to_json_is_serializable(self):
        cfg = self._load_cfg()
        json.dumps(cfg.to_json())

    def test_legacy_run_purer_labeler_preserved(self):
        cfg = self._load_cfg()
        self.assertTrue(cfg.run_purer_labeler)

    def test_legacy_purer_classification_preserved(self):
        cfg = self._load_cfg()
        pc = cfg.purer_classification
        self.assertEqual(pc.backend, "lmstudio")
        self.assertEqual(pc.model, "google/gemma-4-31b")
        self.assertEqual(pc.n_runs, 1)
        self.assertFalse(pc.per_run_models)

    def test_legacy_purer_cue_config_preserved(self):
        cfg = self._load_cfg()
        self.assertFalse(cfg.purer_cue.skip_lesson_content)


# ============================================================================
# Legacy migration tests
# ============================================================================

class TestLegacyMigrationDetection(unittest.TestCase):
    """Verify is_legacy_project() correctly identifies pre-modular projects."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dir(self, *parts):
        d = os.path.join(self.tmpdir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def test_is_legacy_with_master_segments_no_segmented(self):
        self._make_dir("02_meta", "training_data")
        with open(os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"), "w") as f:
            f.write("{}")
        from process.legacy_migration import is_legacy_project
        self.assertTrue(is_legacy_project(self.tmpdir))

    def test_not_legacy_when_segmented_dir_populated(self):
        self._make_dir("02_meta", "training_data")
        with open(os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"), "w") as f:
            f.write("{}")
        self._make_dir("01_transcripts", "segmented", "c1s1")
        with open(os.path.join(self.tmpdir, "01_transcripts", "segmented", "c1s1", "segments.jsonl"), "w") as f:
            f.write("{}")
        from process.legacy_migration import is_legacy_project
        self.assertFalse(is_legacy_project(self.tmpdir))

    def test_not_legacy_when_no_master_segments(self):
        from process.legacy_migration import is_legacy_project
        self.assertFalse(is_legacy_project(self.tmpdir))

    def test_not_legacy_when_neither_present(self):
        from process.legacy_migration import is_legacy_project
        self.assertFalse(is_legacy_project(self.tmpdir))


class TestLegacySegmentMigration(unittest.TestCase):
    """Verify migrate_legacy_segments() extracts per-session frozen segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._setup_master_segments()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dir(self, *parts):
        d = os.path.join(self.tmpdir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _setup_master_segments(self):
        segs = [
            _make_segment(segment_id="c1s1_001", session_id="c1s1", segment_index=0,
                          text="Hello I am Diana", word_count=4),
            _make_segment(segment_id="c1s1_002", session_id="c1s1", segment_index=1,
                          text="I have a lot of pain", word_count=6),
            _make_segment(segment_id="c1s2_001", session_id="c1s2", segment_index=0,
                          text="This is session two", word_count=4),
            _make_segment(segment_id="c1s2_002", session_id="c1s2", segment_index=1,
                          text="More text here", word_count=3),
        ]
        ms_dir = self._make_dir("02_meta", "training_data")
        with open(os.path.join(ms_dir, "master_segments.jsonl"), "w", encoding="utf-8") as f:
            for s in segs:
                rec = {
                    "segment_id": s.segment_id, "trial_id": s.trial_id,
                    "participant_id": s.participant_id, "session_id": s.session_id,
                    "session_number": s.session_number, "cohort_id": s.cohort_id,
                    "session_variant": s.session_variant, "segment_index": s.segment_index,
                    "start_time_ms": s.start_time_ms, "end_time_ms": s.end_time_ms,
                    "total_segments_in_session": 2, "speaker": s.speaker,
                    "text": s.text, "word_count": s.word_count,
                    "speakers_in_segment": None, "session_file": "",
                    "primary_stage": 2, "llm_confidence_primary": 0.85,
                    "llm_justification": "test",
                }
                f.write(json.dumps(rec) + "\n")

    def test_migration_produces_per_session_files(self):
        from process.legacy_migration import migrate_legacy_segments
        n = migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n, 2)
        for sid in ("c1s1", "c1s2"):
            seg_path = os.path.join(
                self.tmpdir, "01_transcripts", "segmented", sid, "segments.jsonl")
            self.assertTrue(os.path.isfile(seg_path), f"Missing {seg_path}")
            meta_path = os.path.join(
                self.tmpdir, "01_transcripts", "segmented", sid, "segmentation_meta.json")
            self.assertTrue(os.path.isfile(meta_path), f"Missing {meta_path}")

    def test_migration_params_hash_is_legacy_sentinel(self):
        from process.legacy_migration import migrate_legacy_segments
        migrate_legacy_segments(self.tmpdir)
        for sid in ("c1s1", "c1s2"):
            meta_path = os.path.join(
                self.tmpdir, "01_transcripts", "segmented", sid, "segmentation_meta.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["params_hash"], "legacy-pre-modular")

    def test_migration_drops_classification_fields(self):
        from process.legacy_migration import migrate_legacy_segments
        migrate_legacy_segments(self.tmpdir)
        seg_path = os.path.join(
            self.tmpdir, "01_transcripts", "segmented", "c1s1", "segments.jsonl")
        with open(seg_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                self.assertNotIn("primary_stage", rec)
                self.assertNotIn("llm_confidence_primary", rec)
                self.assertNotIn("llm_justification", rec)
                self.assertNotIn("rater_votes", rec)

    def test_migration_segment_identity_preserved(self):
        from process.legacy_migration import migrate_legacy_segments
        migrate_legacy_segments(self.tmpdir)
        seg_path = os.path.join(
            self.tmpdir, "01_transcripts", "segmented", "c1s1", "segments.jsonl")
        with open(seg_path, encoding="utf-8") as f:
            lines = [json.loads(l.strip()) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["segment_id"], "c1s1_001")
        self.assertEqual(lines[0]["text"], "Hello I am Diana")
        self.assertEqual(lines[0]["segment_index"], 0)
        self.assertEqual(lines[1]["segment_id"], "c1s1_002")
        self.assertEqual(lines[1]["text"], "I have a lot of pain")
        self.assertEqual(lines[1]["segment_index"], 1)

    def test_migration_idempotent(self):
        from process.legacy_migration import migrate_legacy_segments
        n1 = migrate_legacy_segments(self.tmpdir)
        n2 = migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n1, 2)
        self.assertEqual(n2, 0)

    def test_migration_total_segment_count(self):
        from process.legacy_migration import migrate_legacy_segments
        migrate_legacy_segments(self.tmpdir)
        total = 0
        for sid in ("c1s1", "c1s2"):
            seg_path = os.path.join(
                self.tmpdir, "01_transcripts", "segmented", sid, "segments.jsonl")
            with open(seg_path, encoding="utf-8") as f:
                total += sum(1 for l in f if l.strip())
        self.assertEqual(total, 4)


# ============================================================================
# Segmentation safety tests
# ============================================================================

class TestSegmentationFreshness(unittest.TestCase):
    """Verify is_segmentation_fresh() prevents re-segmentation of legacy projects."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_segmented(self, session_id, params_hash):
        seg_dir = os.path.join(self.tmpdir, "01_transcripts", "segmented", session_id)
        os.makedirs(seg_dir, exist_ok=True)
        with open(os.path.join(seg_dir, "segments.jsonl"), "w") as f:
            f.write("{}")
        with open(os.path.join(seg_dir, "segmentation_meta.json"), "w") as f:
            json.dump({"params_hash": params_hash}, f)

    def test_legacy_pre_modular_is_always_fresh(self):
        self._make_segmented("c1s1", "legacy-pre-modular")
        from process.segments_io import is_segmentation_fresh
        self.assertTrue(is_segmentation_fresh(self.tmpdir, "c1s1", "some-other-hash"))

    def test_current_hash_matches_stored_fresh(self):
        self._make_segmented("c1s1", "abc123")
        from process.segments_io import is_segmentation_fresh
        self.assertTrue(is_segmentation_fresh(self.tmpdir, "c1s1", "abc123"))

    def test_different_hash_not_fresh(self):
        self._make_segmented("c1s1", "abc123")
        from process.segments_io import is_segmentation_fresh
        self.assertFalse(is_segmentation_fresh(self.tmpdir, "c1s1", "xyz789"))

    def test_no_segments_file_not_fresh(self):
        os.makedirs(os.path.join(self.tmpdir, "01_transcripts", "segmented", "c1s1"))
        from process.segments_io import is_segmentation_fresh
        self.assertFalse(is_segmentation_fresh(self.tmpdir, "c1s1", "any"))

    def test_no_meta_file_not_fresh(self):
        seg_dir = os.path.join(self.tmpdir, "01_transcripts", "segmented", "c1s1")
        os.makedirs(seg_dir, exist_ok=True)
        with open(os.path.join(seg_dir, "segments.jsonl"), "w") as f:
            f.write("{}")
        from process.segments_io import is_segmentation_fresh
        self.assertFalse(is_segmentation_fresh(self.tmpdir, "c1s1", "any"))


class TestFrozenArtifactGuards(unittest.TestCase):
    """Verify write_frozen() prevents silent overwrite."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_frozen_raises_on_existing(self):
        from process._freeze import write_frozen, FrozenArtifactError
        path = os.path.join(self.tmpdir, "frozen.txt")
        with open(path, "w") as f:
            f.write("original")
        with self.assertRaises(FrozenArtifactError):
            write_frozen(path, lambda fh: fh.write("new"), force=False)

    def test_write_frozen_succeeds_with_force(self):
        from process._freeze import write_frozen
        path = os.path.join(self.tmpdir, "frozen.txt")
        with open(path, "w") as f:
            f.write("original")
        write_frozen(path, lambda fh: fh.write("new"), force=True)
        with open(path) as f:
            self.assertEqual(f.read(), "new")

    def test_write_frozen_succeeds_when_no_file(self):
        from process._freeze import write_frozen
        path = os.path.join(self.tmpdir, "new.txt")
        write_frozen(path, lambda fh: fh.write("fresh"), force=False)
        with open(path) as f:
            self.assertEqual(f.read(), "fresh")


class TestSegmentReadBack(unittest.TestCase):
    """Verify read_session_segments() faithfully reconstructs segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_segments(self, session_id, segments):
        from process._freeze import write_frozen
        seg_dir = os.path.join(self.tmpdir, "01_transcripts", "segmented", session_id)
        os.makedirs(seg_dir, exist_ok=True)
        seg_path = os.path.join(seg_dir, "segments.jsonl")
        raw_fields = ("segment_id", "trial_id", "participant_id", "session_id",
                      "session_number", "cohort_id", "session_variant", "segment_index",
                      "start_time_ms", "end_time_ms", "total_segments_in_session",
                      "speaker", "text", "word_count", "speakers_in_segment", "session_file")
        def _write(fh):
            for seg in segments:
                rec = {f: getattr(seg, f, None) for f in raw_fields}
                fh.write(json.dumps(rec) + "\n")
        write_frozen(seg_path, _write, force=True)
        meta_path = os.path.join(seg_dir, "segmentation_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"params_hash": "legacy-pre-modular"}, f)

    def test_roundtrip_text_identical(self):
        segs = [_make_segment(text="Original text exactly preserved")]
        self._write_segments("c1s1", segs)
        from process.segments_io import read_session_segments
        restored = read_session_segments(self.tmpdir, "c1s1")
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].text, "Original text exactly preserved")

    def test_roundtrip_segment_id_identical(self):
        segs = [_make_segment(segment_id="c1s7_133")]
        self._write_segments("c1s7", segs)
        from process.segments_io import read_session_segments
        restored = read_session_segments(self.tmpdir, "c1s7")
        self.assertEqual(restored[0].segment_id, "c1s7_133")

    def test_roundtrip_session_id_identical(self):
        segs = [_make_segment(session_id="c2s4")]
        self._write_segments("c2s4", segs)
        from process.segments_io import read_session_segments
        restored = read_session_segments(self.tmpdir, "c2s4")
        self.assertEqual(restored[0].session_id, "c2s4")

    def test_roundtrip_segment_index_identical(self):
        segs = [_make_segment(segment_index=27)]
        self._write_segments("c1s1", segs)
        from process.segments_io import read_session_segments
        restored = read_session_segments(self.tmpdir, "c1s1")
        self.assertEqual(restored[0].segment_index, 27)

    def test_roundtrip_all_core_fields(self):
        seg = _make_segment(
            segment_id="c2s8_289", trial_id="MoveMORE",
            participant_id="Participant_MM014", session_id="c2s8",
            session_number=8, cohort_id=2, session_variant="a",
            segment_index=3, start_time_ms=1162000, end_time_ms=1189000,
            total_segments_in_session=5, speaker="Participant_MM014",
            text="If I calm my mind, it reduces pain significantly",
            word_count=8, session_file="c2s8.json",
        )
        self._write_segments("c2s8", [seg])
        from process.segments_io import read_session_segments
        r = read_session_segments(self.tmpdir, "c2s8")[0]
        self.assertEqual(r.segment_id, "c2s8_289")
        self.assertEqual(r.trial_id, "MoveMORE")
        self.assertEqual(r.participant_id, "Participant_MM014")
        self.assertEqual(r.session_id, "c2s8")
        self.assertEqual(r.session_number, 8)
        self.assertEqual(r.cohort_id, 2)
        self.assertEqual(r.session_variant, "a")
        self.assertEqual(r.segment_index, 3)
        self.assertEqual(r.start_time_ms, 1162000)
        self.assertEqual(r.end_time_ms, 1189000)
        self.assertEqual(r.total_segments_in_session, 5)
        self.assertEqual(r.speaker, "Participant_MM014")
        self.assertEqual(r.text, "If I calm my mind, it reduces pain significantly")
        self.assertEqual(r.word_count, 8)
        self.assertEqual(r.session_file, "c2s8.json")

    def test_classification_fields_default_on_readback(self):
        segs = [_make_segment(text="test", primary_stage=2, llm_confidence_primary=0.9)]
        self._write_segments("c1s1", segs)
        from process.segments_io import read_session_segments
        r = read_session_segments(self.tmpdir, "c1s1")[0]
        self.assertIsNone(r.primary_stage)
        self.assertIsNone(r.llm_confidence_primary)


# ============================================================================
# Testset locking and refresh tests
# ============================================================================

class TestTestsetLocking(unittest.TestCase):
    """Verify the create/refresh/verify cycle for frozen testsets."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._mkdirs = lambda *p: os.makedirs(os.path.join(self.tmpdir, *p), exist_ok=True)
        self._setup_project()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup_project(self):
        self._segments = [
            _make_segment(segment_id="c1s1_001", session_id="c1s1", segment_index=0,
                          text="I am Diana and I have pain", word_count=7,
                          participant_id="Participant_MM001",
                          primary_stage=0, llm_confidence_primary=0.9,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 0, "confidence": 0.9}]),
            _make_segment(segment_id="c1s1_002", session_id="c1s1", segment_index=1,
                          text="Breathing helps me refocus", word_count=4,
                          participant_id="Participant_MM001",
                          primary_stage=2, llm_confidence_primary=0.85,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 2, "confidence": 0.85}]),
            _make_segment(segment_id="c1s2_001", session_id="c1s2", segment_index=0,
                          text="Check-in from session two", word_count=4,
                          participant_id="Participant_MM005",
                          primary_stage=0, llm_confidence_primary=0.75,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 0, "confidence": 0.75}]),
        ]
        from process.segments_io import write_session_segments
        for sid in ("c1s1", "c1s2"):
            write_session_segments(self.tmpdir, sid,
                                   [s for s in self._segments if s.session_id == sid],
                                   "legacy-pre-modular")

    def _mock_framework(self):
        m = MagicMock()
        m.name = "vaamr"
        m.version = "2.0"
        m.themes = [MagicMock(theme_id=i, short_name=n) for i, n in
                    enumerate(["Vigilance", "Avoidance", "Attention", "Metacognition", "Reappraisal"])]
        return m

    def test_create_frozen_testset_writes_all_artifacts(self):
        from process.assembly.human_forms import create_frozen_testset
        from process import output_paths as _paths
        mock_fw = self._mock_framework()
        create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            name="vaamr_testset_1", kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None, force=True)
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)),
                        "Missing human worksheet #1")
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)),
                        "Missing AI answer key #1")
        # No subdirectory, no manifest, no snapshot
        ts_dir = _paths.testsets_dir(self.tmpdir)
        self.assertFalse(
            any(os.path.isdir(os.path.join(ts_dir, e)) for e in os.listdir(ts_dir)),
            "No subdirectory should exist"
        )

    def test_human_worksheet_has_no_classification_fields(self):
        from process.assembly.human_forms import create_frozen_testset
        from process import output_paths as _paths
        create_frozen_testset(
            self._segments, self._mock_framework(), self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        with open(_paths.testset_human_flat_path(self.tmpdir, 1)) as f:
            content = f.read()
        self.assertNotIn("RATER BALLOTS", content)
        self.assertNotIn("CONSENSUS:", content)
        self.assertNotIn("Mean confidence", content)
        self.assertIn("Primary stage: ___", content)

    def test_ai_answer_key_has_classifications(self):
        from process.assembly.human_forms import create_frozen_testset
        from process import output_paths as _paths
        create_frozen_testset(
            self._segments, self._mock_framework(), self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        with open(_paths.testset_ai_flat_path(self.tmpdir, 1)) as f:
            content = f.read()
        self.assertIn("RATER BALLOTS", content)
        self.assertIn("CONSENSUS:", content)

    def test_create_frozen_testset_sequential_numbering(self):
        from process.assembly.human_forms import create_frozen_testset
        from process import output_paths as _paths
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"; mock_fw.themes = []
        p1 = create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        p2 = create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        self.assertEqual(p1, _paths.testset_human_flat_path(self.tmpdir, 1))
        self.assertEqual(p2, _paths.testset_human_flat_path(self.tmpdir, 2))

    def test_refresh_updates_classifications_preserves_text(self):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        from process import output_paths as _paths
        mock_fw = self._mock_framework()
        create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        # Re-classify
        for seg in self._segments:
            seg.primary_stage = 3
            seg.llm_confidence_primary = 0.95
            seg.llm_justification = "Updated justification"
            seg.secondary_stage = 2
            seg.llm_confidence_secondary = 0.70
        refresh_testset_answer_key(
            {s.segment_id: s for s in self._segments}, mock_fw, self.tmpdir,
            1, codebook_enabled=False, codebook=None)
        with open(_paths.testset_ai_flat_path(self.tmpdir, 1)) as f:
            content = f.read()
        # AI key shows updated consensus stage (Metacognition = stage 3) and updated confidence
        self.assertIn("Metacognition", content)
        self.assertIn("0.95", content)

    def test_refresh_with_missing_segment_raises(self):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        from process._freeze import FrozenArtifactError
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"; mock_fw.themes = []
        create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key(
                {}, mock_fw, self.tmpdir, 1,
                codebook_enabled=False, codebook=None)

    def test_refresh_detects_missing_segments(self):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        from process._freeze import FrozenArtifactError
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"; mock_fw.themes = []
        create_frozen_testset(
            self._segments, mock_fw, self.tmpdir,
            name="vaamr_testset_1", kind="vaamr", n_sets=2, set_index=1,
            fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None, force=True)
        segs_by_id = {s.segment_id: s for s in self._segments if s.segment_id != "c1s1_001"}
        # Missing segments are detected as SHA mismatch (None → False) or not-found on second pass
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key(segs_by_id, mock_fw, self.tmpdir,
                                       "vaamr_testset_1", codebook_enabled=False, codebook=None)

    def test_sha_verify_matches(self):
        from process._freeze import verify_content_sha, sha256_text
        ts_dir = os.path.join(self.tmpdir, "04_validation", "testsets", "test")
        os.makedirs(ts_dir, exist_ok=True)
        snap_path = os.path.join(ts_dir, "segments_snapshot.jsonl")
        seg = _make_segment(text="Original text")
        with open(snap_path, "w") as f:
            f.write(json.dumps({"segment_id": seg.segment_id, "content_sha256": sha256_text(seg.text)}) + "\n")
        self.assertTrue(verify_content_sha(snap_path, {seg.segment_id: seg}).get(seg.segment_id))

    def test_sha_verify_mismatch(self):
        from process._freeze import verify_content_sha, sha256_text
        ts_dir = os.path.join(self.tmpdir, "04_validation", "testsets", "test")
        os.makedirs(ts_dir, exist_ok=True)
        snap_path = os.path.join(ts_dir, "segments_snapshot.jsonl")
        seg = _make_segment(text="Original text")
        with open(snap_path, "w") as f:
            f.write(json.dumps({"segment_id": seg.segment_id, "content_sha256": sha256_text("Original text")}) + "\n")
        seg.text = "CHANGED"
        self.assertFalse(verify_content_sha(snap_path, {seg.segment_id: seg}).get(seg.segment_id, True))

    def test_create_missing_false_skips(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"; mock_fw.themes = []
        ts_cfg = TestSetsConfig(vaamr=TestSetSpec(enabled=True, name="vaamr_testset", n_sets=2))
        dirs = generate_or_refresh_validation_testsets(
            self._segments, mock_fw, self.tmpdir, test_sets_config=ts_cfg, create_missing=False)
        self.assertEqual(dirs, [])
        # Create one, then refresh should find it
        from process.assembly.human_forms import create_frozen_testset
        create_frozen_testset(
            self._segments, mock_fw, self.tmpdir, name="vaamr_testset_1",
            kind="vaamr", n_sets=2, set_index=1, fraction_per_set=0.1, random_seed=42,
            codebook_enabled=False, codebook=None, force=True)
        dirs = generate_or_refresh_validation_testsets(
            self._segments, mock_fw, self.tmpdir, test_sets_config=ts_cfg, create_missing=False)
        self.assertEqual(len(dirs), 1)

    def test_create_missing_true_creates(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"
        mock_fw.themes = [MagicMock(theme_id=0, short_name="Vigilance")]
        ts_cfg = TestSetsConfig(vaamr=TestSetSpec(enabled=True, name="vaamr_testset", n_sets=1))
        dirs = generate_or_refresh_validation_testsets(
            self._segments, mock_fw, self.tmpdir, test_sets_config=ts_cfg, create_missing=True)
        self.assertEqual(len(dirs), 1)


# ============================================================================
# Overlay I/O safety tests
# ============================================================================

class TestOverlayIOSafety(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, "02_meta", "classifications"), exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_overlay_lines(self, key):
        path = os.path.join(self.tmpdir, "02_meta", "classifications", f"{key}_labels.jsonl")
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(l.strip()) for l in f if l.strip()]

    def test_theme_overlay_fields_exclude_core_identity(self):
        from process.classifications_io import THEME_OVERLAY_FIELDS
        forbidden = {"segment_id", "text", "session_id", "segment_index",
                     "trial_id", "participant_id", "speaker", "word_count",
                     "start_time_ms", "end_time_ms"}
        for field in THEME_OVERLAY_FIELDS:
            self.assertNotIn(field, forbidden, f"Core field '{field}' leaked into THEME_OVERLAY_FIELDS")

    def test_purer_overlay_fields_exclude_core_identity(self):
        from process.classifications_io import PURER_OVERLAY_FIELDS
        forbidden = {"segment_id", "text", "session_id", "segment_index",
                     "trial_id", "participant_id", "speaker", "word_count"}
        for field in PURER_OVERLAY_FIELDS:
            self.assertNotIn(field, forbidden, f"Core field '{field}' leaked into PURER_OVERLAY_FIELDS")

    def test_codebook_overlay_fields_exclude_core_identity(self):
        from process.classifications_io import CODEBOOK_OVERLAY_FIELDS
        forbidden = {"segment_id", "text", "session_id", "segment_index",
                     "trial_id", "participant_id", "speaker", "word_count"}
        for field in CODEBOOK_OVERLAY_FIELDS:
            self.assertNotIn(field, forbidden, f"Core field '{field}' leaked into CODEBOOK_OVERLAY_FIELDS")

    def test_write_then_read_theme_overlay(self):
        from process.classifications_io import write_theme_overlay, apply_theme_overlay
        segments = [
            _make_segment(segment_id="c1s1_001", text="text 1",
                          primary_stage=2, llm_confidence_primary=0.85,
                          llm_justification="test just", needs_review=False,
                          agreement_level="majority", agreement_fraction=0.67,
                          consensus_vote="CODED", tie_broken_by_confidence=False,
                          llm_run_consistency=3),
        ]
        write_theme_overlay(self.tmpdir, segments)
        lines = self._read_overlay_lines("theme")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["segment_id"], "c1s1_001")
        self.assertEqual(lines[0]["primary_stage"], 2)
        self.assertEqual(lines[0]["llm_confidence_primary"], 0.85)
        # Read back
        fresh = [_make_segment(segment_id="c1s1_001", text="text 1")]
        apply_theme_overlay(self.tmpdir, {s.segment_id: s for s in fresh})
        self.assertEqual(fresh[0].primary_stage, 2)
        self.assertEqual(fresh[0].llm_confidence_primary, 0.85)
        self.assertEqual(fresh[0].text, "text 1")

    def test_overlay_records_sorted_by_segment_id(self):
        from process.classifications_io import write_theme_overlay
        segments = [
            _make_segment(segment_id="c1s2_001", primary_stage=0),
            _make_segment(segment_id="c1s1_003", primary_stage=1),
            _make_segment(segment_id="c1s1_001", primary_stage=2),
        ]
        write_theme_overlay(self.tmpdir, segments)
        lines = self._read_overlay_lines("theme")
        ids = [r["segment_id"] for r in lines]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, ["c1s1_001", "c1s1_003", "c1s2_001"])


# ============================================================================
# Master segments compatibility
# ============================================================================

class TestMasterSegmentsCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_master_segments(self, records):
        path = os.path.join(self.tmpdir, "02_meta", "training_data")
        os.makedirs(path, exist_ok=True)
        fp = os.path.join(path, "master_segments.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return fp

    def test_parse_working_branch_format(self):
        rec = {
            "segment_id": "c1s1_002", "trial_id": "MoveMORE",
            "participant_id": "Participant_MM001", "session_id": "c1s1",
            "session_number": 1, "cohort_id": 1, "session_variant": "",
            "segment_index": 1, "start_time_ms": 5269000, "end_time_ms": 5277000,
            "total_segments_in_session": 15, "speaker": "Participant_MM001",
            "text": "I would like to avoid surgery", "word_count": 158,
            "speakers_in_segment": None, "session_file": "c1s1.json",
            "primary_stage": 2, "secondary_stage": None,
            "llm_confidence_primary": 0.85, "llm_confidence_secondary": None,
            "llm_justification": "test",
            "rater_ids": ["A", "B", "C"],
            "rater_votes": json.dumps([{"rater": "A", "vote": "CODED"}]),
            "agreement_level": "majority", "agreement_fraction": 0.67,
            "needs_review": False, "consensus_vote": "CODED",
            "tie_broken_by_confidence": False, "llm_run_consistency": 3,
            "codebook_labels_embedding": None, "codebook_labels_llm": None,
            "codebook_labels_ensemble": None, "codebook_disagreements": None,
            "human_label": None, "human_secondary_label": None,
            "adjudicated_label": None, "in_human_coded_subset": False,
            "label_status": "llm_only", "final_label": 2,
            "final_label_source": "llm_consensus", "label_confidence_tier": "high",
        }
        self._write_master_segments([rec])
        from process.segments_io import _load_segments_from_jsonl
        segs = _load_segments_from_jsonl(
            os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"))
        self.assertEqual(len(segs), 1)
        s = segs[0]
        self.assertEqual(s.segment_id, "c1s1_002")
        self.assertEqual(s.text, "I would like to avoid surgery")
        self.assertEqual(s.primary_stage, 2)
        self.assertEqual(s.llm_confidence_primary, 0.85)
        self.assertIsNotNone(s.rater_votes)

    def test_parse_handles_missing_optional_fields(self):
        rec = {"segment_id": "c1s1_001", "session_id": "c1s1", "segment_index": 0, "text": "minimal"}
        self._write_master_segments([rec])
        from process.segments_io import _load_segments_from_jsonl
        segs = _load_segments_from_jsonl(
            os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"))
        self.assertEqual(segs[0].text, "minimal")

    def test_parse_handles_nan_values(self):
        rec = {"segment_id": "c1s1_001", "session_id": "c1s1", "segment_index": 0,
               "text": "text", "primary_stage": None, "llm_confidence_primary": "nan", "cohort_id": "nan"}
        self._write_master_segments([rec])
        from process.segments_io import _load_segments_from_jsonl
        segs = _load_segments_from_jsonl(
            os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"))
        self.assertIsNone(segs[0].primary_stage)
        self.assertIsNone(segs[0].llm_confidence_primary)


# ============================================================================
# Integration: End-to-end legacy import safe path
# ============================================================================

class TestEndToEndLegacyImport(unittest.TestCase):
    """Integration test: simulate the full legacy-import → reclassify → refresh cycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mkdirs(self, *parts):
        d = os.path.join(self.tmpdir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def test_full_legacy_import_cycle(self):
        """
        Full cycle:
        1. Working-branch project exists (master_segments + flat testsets)
        2. Beta code detects legacy, migrates segments + testsets
        3. Re-classification writes new overlays (theme, purer, codebook)
        4. Assemble joins frozen + overlays
        5. Refresh updates AI answer keys with new classifications
        6. Verify human worksheet content unchanged
        """
        # Step 1: Working-branch project
        segs = [
            _make_segment(segment_id="c1s1_002", session_id="c1s1", segment_index=1,
                          text="I would like to avoid surgery and have less pain",
                          word_count=10, participant_id="Participant_MM001",
                          primary_stage=1, llm_confidence_primary=0.85,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 1, "confidence": 0.85}]),
            _make_segment(segment_id="c1s1_012", session_id="c1s1", segment_index=11,
                          text="I enjoyed the sitting posture",
                          word_count=5, participant_id="Participant_MM001",
                          primary_stage=2, llm_confidence_primary=0.90,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 2, "confidence": 0.90}]),
            _make_segment(segment_id="c2s3_180", session_id="c2s3", segment_index=5,
                          text="I'm seeing results already with mindfulness",
                          word_count=7, participant_id="Participant_MM014",
                          primary_stage=4, llm_confidence_primary=0.88,
                          rater_votes=[{"rater": "A", "vote": "CODED", "stage": 4, "confidence": 0.88}]),
        ]
        ms_dir = self._mkdirs("02_meta", "training_data")
        with open(os.path.join(ms_dir, "master_segments.jsonl"), "w", encoding="utf-8") as f:
            for s in segs:
                rec = {"segment_id": s.segment_id, "trial_id": s.trial_id,
                       "participant_id": s.participant_id, "session_id": s.session_id,
                       "session_number": s.session_number, "cohort_id": s.cohort_id,
                       "session_variant": s.session_variant or "",
                       "segment_index": s.segment_index,
                       "start_time_ms": s.start_time_ms, "end_time_ms": s.end_time_ms,
                       "total_segments_in_session": 15 if s.session_id == "c1s1" else 10,
                       "speaker": s.speaker, "text": s.text, "word_count": s.word_count,
                       "speakers_in_segment": s.speakers_in_segment,
                       "session_file": f"{s.session_id}.json",
                       "primary_stage": s.primary_stage,
                       "llm_confidence_primary": s.llm_confidence_primary,
                       "llm_justification": s.llm_justification,
                       "rater_votes": json.dumps(s.rater_votes) if s.rater_votes else None,
                       "agreement_level": "single", "agreement_fraction": 1.0,
                       "needs_review": False, "consensus_vote": "CODED",
                       "tie_broken_by_confidence": False, "llm_run_consistency": 1,
                       "secondary_stage": None, "llm_confidence_secondary": None,
                       "rater_ids": None,
                       "codebook_labels_embedding": None, "codebook_labels_llm": None,
                       "codebook_labels_ensemble": None, "codebook_disagreements": None,
                       "human_label": None, "human_secondary_label": None,
                       "adjudicated_label": None, "in_human_coded_subset": False,
                       "label_status": "llm_only", "final_label": s.primary_stage,
                       "final_label_source": "llm_consensus", "label_confidence_tier": "high"}
                f.write(json.dumps(rec) + "\n")

        # Flat testset worksheets
        ts_flat = self._mkdirs("04_validation", "testsets")
        human_text = (
            "==============================================================================\n"
            "VALIDATION TEST SET 1 of 2 — HUMAN CODING WORKSHEET\n"
            "==============================================================================\n"
            "Generated: 2026-05-03   Segments: 2\n"
            "Instructions: Code each segment independently.\n"
            "==============================================================================\n"
            "[ITEM 001]  Session: c1s1   Segment 002\n"
            "            02:26:09–02:27:17   10w   Participant: Participant_MM001\n"
            "------------------------------------------------------------------------------\n"
            "  I would like to avoid surgery and have less pain\n\n"
            "  Primary stage: ___   Secondary (optional): ___\n"
            "  Rationale: ____________________________________________________________\n"
            "\n"
            "==============================================================================\n"
            "[ITEM 002]  Session: c2s3   Segment 006\n"  # segment_index=5 → 1-based=6
            "            00:02:15–00:16:06   7w   Participant: Participant_MM014\n"
            "------------------------------------------------------------------------------\n"
            "  I'm seeing results already with mindfulness\n\n"
            "  Primary stage: ___   Secondary (optional): ___\n"
            "  Rationale: ____________________________________________________________\n"
        )
        with open(os.path.join(ts_flat, "human_classification_testset_worksheet_1.txt"), "w") as f:
            f.write(human_text)
        with open(os.path.join(ts_flat, "AI_classification_testset_worksheet_1.txt"), "w") as f:
            f.write("OLD AI CLASSIFICATIONS")

        # Step 2: Legacy detection
        from process.legacy_migration import is_legacy_project
        self.assertTrue(is_legacy_project(self.tmpdir))

        # Step 3: Migrate segments (v2.0 → frozen per-session segments)
        from process.legacy_migration import migrate_legacy_segments
        self.assertEqual(migrate_legacy_segments(self.tmpdir), 2)

        # Step 4: Verify segments match original
        from process.segments_io import read_session_segments
        for sid in ("c1s1", "c2s3"):
            for rs in read_session_segments(self.tmpdir, sid):
                orig = next(s for s in segs if s.segment_id == rs.segment_id)
                self.assertEqual(rs.text, orig.text, f"Text mismatch for {rs.segment_id}")
                self.assertEqual(rs.segment_index, orig.segment_index, f"Index mismatch for {rs.segment_id}")

        # Step 5: Re-classify — write new overlays
        for seg in segs:
            seg.primary_stage = 3
            seg.llm_confidence_primary = 0.92
            seg.llm_justification = "New justification after framework update"
            seg.secondary_stage = 2
            seg.llm_confidence_secondary = 0.65
        from process.classifications_io import write_theme_overlay
        write_theme_overlay(self.tmpdir, segs)

        # Step 6: Verify frozen segments unchanged
        for rs in read_session_segments(self.tmpdir, "c1s1"):
            self.assertIsNone(rs.primary_stage)

        # Step 7: Refresh testset AI answer key
        mock_fw = MagicMock()
        mock_fw.name = "vaamr"; mock_fw.version = "3.0"
        mock_fw.themes = [MagicMock(theme_id=1, short_name="Avoidance"),
                          MagicMock(theme_id=2, short_name="Attention"),
                          MagicMock(theme_id=3, short_name="Metacognition")]
        from process.assembly.human_forms import refresh_testset_answer_key
        from process import output_paths as _paths
        # Worksheet was created as #1 (first flat testset)
        refresh_testset_answer_key(
            {s.segment_id: s for s in segs}, mock_fw, self.tmpdir,
            1, codebook_enabled=False, codebook=None)

        # Step 8: Verify AI key updated — shows new consensus stage, not old rater ballot stage
        with open(_paths.testset_ai_flat_path(self.tmpdir, 1)) as f:
            ai_content = f.read()
        self.assertIn("Metacognition", ai_content)
        self.assertIn("0.92", ai_content)  # updated confidence

        # Step 9: Verify human worksheet unchanged
        with open(_paths.testset_human_flat_path(self.tmpdir, 1)) as f:
            human_content_migrated = f.read()
        self.assertIn("I would like to avoid surgery and have less pain", human_content_migrated)
        self.assertIn("I'm seeing results already with mindfulness", human_content_migrated)
        self.assertIn("Primary stage: ___", human_content_migrated)
        self.assertNotIn("Metacognition", human_content_migrated)
        self.assertNotIn("New justification", human_content_migrated)

        # Step 10: Item ordering parity
        pattern = re.compile(r'\[ITEM (\d+)\].*?Session: (\S+)\s+Segment (\d+)')
        self.assertEqual(pattern.findall(human_content_migrated), pattern.findall(ai_content))

    def test_master_segments_load_after_migration(self):
        self._mkdirs("02_meta", "training_data")
        with open(os.path.join(self.tmpdir, "02_meta", "training_data", "master_segments.jsonl"), "w") as f:
            f.write(json.dumps({"segment_id": "test_001", "session_id": "test", "segment_index": 0, "text": "hello"}) + "\n")
        from process.segments_io import read_master_segments
        self.assertEqual(read_master_segments(self.tmpdir)[0].text, "hello")


# ============================================================================
# CLI surface — legacy-safe commands
# ============================================================================

class TestCLILegacySafeCommands(unittest.TestCase):
    def _subcommands(self):
        from qra import _build_parser
        parser, _ts, _cv = _build_parser()
        return list(parser._subparsers._actions[1].choices.keys())

    def test_ingest_registered(self):
        self.assertIn("ingest", self._subcommands())

    def test_classify_registered(self):
        self.assertIn("classify", self._subcommands())

    def test_assemble_registered(self):
        self.assertIn("assemble", self._subcommands())

    def test_validate_registered(self):
        self.assertIn("validate", self._subcommands())

    def test_analyze_registered(self):
        self.assertIn("analyze", self._subcommands())

    def test_testset_registered(self):
        self.assertIn("testset", self._subcommands())

    def test_build_config_safe_with_partial_args(self):
        import qra
        ns = MagicMock()
        ns.config = None
        ns.output_dir = "./test_output"
        ns.transcript_dir = "./test_input"
        ns.trial_id = "test"
        ns.run_codebook_classifier = False
        ns.run_purer_labeler = True
        ns.auto_analyze = False
        del ns.backend
        del ns.model
        del ns.n_runs
        try:
            qra._build_config(ns)
        except Exception as e:
            self.fail(f"_build_config failed with minimal namespace: {e}")


class TestTestsetRefreshLegacyNotSkipped(unittest.TestCase):
    """Regression: `testset refresh` regenerates legacy AI keys instead of skipping them.

    Previously legacy (legacy_import=true) testsets were short-circuited to a header-only
    sync. They must now flow into the refresh engine so re-run models update the AI key,
    while the human worksheet stays frozen.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        ts = os.path.join(self.tmpdir, "04_validation", "testsets")
        os.makedirs(ts, exist_ok=True)
        # Minimal human worksheet so the refresh loop discovers testset #1.
        with open(os.path.join(ts, "human_classification_testset_worksheet_1.txt"), "w") as f:
            f.write("VALIDATION TEST SET 1 of 1 — HUMAN CODING WORKSHEET\n")
        # Mark it as an imported legacy testset.
        meta = os.path.join(self.tmpdir, "02_meta", "testset_meta")
        os.makedirs(meta, exist_ok=True)
        with open(os.path.join(meta, "human_classification_testset_worksheet_1.meta.json"), "w") as f:
            json.dump({"legacy_import": True, "segments": []}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_legacy_testset_flows_into_refresh_engine(self):
        from types import SimpleNamespace
        import qra
        from process import output_paths as _paths
        ai_path = _paths.testset_ai_flat_path(self.tmpdir, 1)
        ns = SimpleNamespace(output_dir=self.tmpdir, framework=None, force=False)
        with patch.object(qra, "_load_segments_from_output", return_value=[]), \
             patch.object(qra, "_load_framework", return_value=None), \
             patch("process.assembly.human_forms.refresh_testset_answer_key",
                   return_value=ai_path) as mock_refresh:
            qra.cmd_testset_refresh(ns)
        refreshed_nums = [c.args[3] for c in mock_refresh.call_args_list]
        self.assertIn(1, refreshed_nums,
                      "Legacy testset #1 must be refreshed, not skipped")
        # force=False must be forwarded so drift stays a hard error.
        self.assertFalse(mock_refresh.call_args_list[0].kwargs["force"])


# ============================================================================
# Framework integrity tests
# ============================================================================

class TestFrameworkIntegrityForLegacy(unittest.TestCase):
    def test_vaamr_loads(self):
        from theme_framework.vaamr import get_vaamr_framework
        fw = get_vaamr_framework()
        self.assertTrue(fw.name.upper().startswith("VAAMR"))

    def test_vaamr_five_stages(self):
        from theme_framework.vaamr import get_vaamr_framework
        fw = get_vaamr_framework()
        self.assertEqual(len(fw.themes), 5)
        self.assertEqual(sorted(t.theme_id for t in fw.themes), [0, 1, 2, 3, 4])

    def test_purer_loads(self):
        from theme_framework.purer import get_purer_framework
        fw = get_purer_framework()
        self.assertTrue(fw.name.upper().startswith("PURER"))

    def test_purer_five_stages(self):
        from theme_framework.purer import get_purer_framework
        fw = get_purer_framework()
        self.assertEqual(len(fw.themes), 5)

    def test_vaamr_theme_ids_stable(self):
        from theme_framework.vaamr import get_vaamr_framework
        by_id = {t.theme_id: t.short_name for t in get_vaamr_framework().themes}
        self.assertEqual(by_id[0], "Vigilance")
        self.assertEqual(by_id[1], "Avoidance")
        self.assertEqual(by_id[2], "Attention Regulation")
        self.assertEqual(by_id[3], "Metacognition")
        self.assertEqual(by_id[4], "Reappraisal")


# ============================================================================
# Regression: known bugs we must guard against
# ============================================================================

class TestRegressionGuards(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_segment_index_not_renumbered_during_readback(self):
        from process.segments_io import write_session_segments, read_session_segments
        seg = _make_segment(segment_index=138)
        write_session_segments(self.tmpdir, "c1s1", [seg], "legacy-pre-modular")
        self.assertEqual(read_session_segments(self.tmpdir, "c1s1")[0].segment_index, 138)

    def test_zero_based_vs_one_based_no_collision(self):
        from process.legacy_migration import _parse_worksheet_items
        ts_dir = os.path.join(self.tmpdir, "04_validation", "testsets")
        os.makedirs(ts_dir, exist_ok=True)
        with open(os.path.join(ts_dir, "test_ws.txt"), "w") as f:
            f.write("[ITEM 001]  Session: c1s1   Segment 002\n")
            f.write("[ITEM 002]  Session: c1s7   Segment 133\n")
        self.assertEqual(_parse_worksheet_items(os.path.join(ts_dir, "test_ws.txt")),
                         [("c1s1", 2), ("c1s7", 133)])

    def test_legacy_config_does_not_trigger_resegmentation(self):
        from process.config import PipelineConfig
        from process.segments_io import is_segmentation_fresh, write_session_segments, params_hash
        from qra import _flatten_wizard_config
        seg = _make_segment()
        write_session_segments(self.tmpdir, "c1s1", [seg], "legacy-pre-modular")
        raw = TestLegacyConfigRoundtrip.legacy_config_dict()
        flat = _flatten_wizard_config(raw)
        cfg = PipelineConfig.from_json(flat)
        current = params_hash(cfg.segmentation)
        self.assertTrue(is_segmentation_fresh(self.tmpdir, "c1s1", current))

    def test_human_worksheet_not_overwritten_on_refresh(self):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        from process.segments_io import write_session_segments
        from process import output_paths as _paths
        mock_fw = MagicMock(); mock_fw.name = "vaamr"; mock_fw.version = "2.0"; mock_fw.themes = []
        seg = _make_segment(primary_stage=0)
        write_session_segments(self.tmpdir, "c1s1", [seg], "legacy-pre-modular")
        create_frozen_testset(
            [seg], mock_fw, self.tmpdir, kind="vaamr",
            n_sets=1, set_index=1, fraction_per_set=1.0, random_seed=42,
            codebook_enabled=False, codebook=None)
        hw_path = _paths.testset_human_flat_path(self.tmpdir, 1)
        with open(hw_path) as f:
            original_human = f.read()
        seg.primary_stage = 4
        refresh_testset_answer_key(
            {seg.segment_id: seg}, mock_fw, self.tmpdir, 1,
            codebook_enabled=False, codebook=None)
        with open(hw_path) as f:
            self.assertEqual(original_human, f.read(),
                             "Human worksheet was modified during refresh!")

    def test_segmentation_not_re_run_for_all_sessions(self):
        """Regression: stage_ingest must NOT re-segment sessions already migrated."""
        from process.segments_io import write_session_segments, read_session_segments
        orig = _make_segment(segment_id="c1s1_001", text="original text from legacy")
        write_session_segments(self.tmpdir, "c1s1", [orig], "legacy-pre-modular")
        # Simulate fresh check
        from process.segments_io import is_segmentation_fresh
        self.assertTrue(is_segmentation_fresh(self.tmpdir, "c1s1", "anything-new"))
        restored = read_session_segments(self.tmpdir, "c1s1")
        self.assertEqual(restored[0].text, "original text from legacy")
