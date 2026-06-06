"""
tests/unit/test_speaker_anonymization.py
-----------------------------------------
Unit tests for process/speaker_anonymization.py.

Covers:
  - load_speaker_map: reads a well-formed key file and returns the correct
    (existing_speaker_map, use_unknown_prefix) tuple
  - Stability across reload: given the same key file on disk, two successive
    calls produce identical speaker maps
  - Imported key path (config.speaker_anonymization_key_path): used on first
    run when no local key file exists yet
  - Missing key file: returns empty map without raising
  - Malformed JSON: returns empty map without raising
  - use_unknown_prefix flag: True when key was seeded from an imported file
    AND a local key also exists; True when only imported path is used
  - New speakers: if a speaker not in the key is encountered, the caller
    (orchestrator) handles adding them — here we test that load_speaker_map
    does not fabricate new entries for missing speakers
  - Key file roundtrip: write a key file in the expected format, reload, and
    assert all fields survive the round-trip intact
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process.speaker_anonymization import load_speaker_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_key(path: str, entries: dict):
    """
    Write a speaker_anonymization_key.json in the expected on-disk format:
      { raw_name: { 'role': str, 'anonymized_id': str } }
    """
    with open(path, 'w') as f:
        json.dump(entries, f)


class _Cfg:
    """Minimal mock of PipelineConfig for speaker_anonymization tests."""
    def __init__(self, key_path=None):
        self.speaker_anonymization_key_path = key_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadSpeakerMap(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- Basic loading ----

    def test_returns_tuple_of_two(self):
        result = load_speaker_map(self.tmp, _Cfg())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_no_key_file_no_config_returns_empty_map(self):
        speaker_map, use_prefix = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(speaker_map, {})
        self.assertFalse(use_prefix)

    def test_missing_key_file_with_nonexistent_config_path_returns_empty(self):
        cfg = _Cfg(key_path='/nonexistent/path/key.json')
        speaker_map, use_prefix = load_speaker_map(self.tmp, cfg)
        self.assertEqual(speaker_map, {})
        self.assertFalse(use_prefix)

    # ---- Loading from meta/ local key ----

    def test_local_key_loaded_correctly(self):
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        entries = {
            'Alice Smith': {'role': 'participant', 'anonymized_id': 'P001'},
            'Dr. Jones':   {'role': 'therapist',   'anonymized_id': 'T001'},
        }
        _write_key(key_path, entries)
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertIn('Alice Smith', speaker_map)
        self.assertIn('Dr. Jones', speaker_map)

    def test_local_key_role_and_id_parsed(self):
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Bob': {'role': 'participant', 'anonymized_id': 'P_007'},
        })
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        role, anon_id = speaker_map['Bob']
        self.assertEqual(role, 'participant')
        self.assertEqual(anon_id, 'P_007')

    def test_local_key_returns_correct_map_format(self):
        """Each value in the map must be a (role, anonymized_id) tuple."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Carol': {'role': 'therapist', 'anonymized_id': 'T_003'},
        })
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        val = speaker_map['Carol']
        self.assertEqual(len(val), 2)
        self.assertEqual(val[0], 'therapist')
        self.assertEqual(val[1], 'T_003')

    # ---- Stability across reload ----

    def test_stability_across_two_loads(self):
        """Two successive loads of the same key file return identical maps."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Dana': {'role': 'participant', 'anonymized_id': 'P_042'},
            'Evan': {'role': 'therapist',   'anonymized_id': 'T_002'},
        })
        map1, prefix1 = load_speaker_map(self.tmp, _Cfg())
        map2, prefix2 = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(map1, map2)
        self.assertEqual(prefix1, prefix2)

    def test_same_input_same_id_after_reload(self):
        """The anonymized_id for a given speaker is stable across reloads."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Frank': {'role': 'participant', 'anonymized_id': 'P_99'},
        })
        map1, _ = load_speaker_map(self.tmp, _Cfg())
        map2, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(map1['Frank'][1], map2['Frank'][1])

    # ---- New speakers are not fabricated ----

    def test_load_does_not_add_new_entries(self):
        """load_speaker_map only reads; it never writes new entries to the map."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Grace': {'role': 'participant', 'anonymized_id': 'P_010'},
        })
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertNotIn('NewPerson', speaker_map)
        self.assertEqual(len(speaker_map), 1)

    # ---- Imported key path (first run, no local key) ----

    def test_imported_key_path_used_when_no_local_key(self):
        """When no local key exists, config.speaker_anonymization_key_path is used."""
        import_dir = tempfile.mkdtemp()
        try:
            imported_key = os.path.join(import_dir, 'imported_key.json')
            _write_key(imported_key, {
                'Hank': {'role': 'participant', 'anonymized_id': 'P_011'},
            })
            cfg = _Cfg(key_path=imported_key)
            speaker_map, use_prefix = load_speaker_map(self.tmp, cfg)
            self.assertIn('Hank', speaker_map)
            self.assertTrue(use_prefix)
        finally:
            shutil.rmtree(import_dir, ignore_errors=True)

    def test_local_key_takes_priority_over_config_path(self):
        """
        When a local key exists AND config.speaker_anonymization_key_path is set,
        the LOCAL key takes priority (stability on re-runs).  use_unknown_prefix
        becomes True because speaker_anonymization_key_path is non-null.
        """
        # Write local key
        local_key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(local_key_path, {
            'Iris': {'role': 'participant', 'anonymized_id': 'P_LOCAL'},
        })

        # Write separate imported key with DIFFERENT content
        import_dir = tempfile.mkdtemp()
        try:
            imported_key = os.path.join(import_dir, 'imported.json')
            _write_key(imported_key, {
                'Jake': {'role': 'therapist', 'anonymized_id': 'T_IMPORT'},
            })
            cfg = _Cfg(key_path=imported_key)
            speaker_map, use_prefix = load_speaker_map(self.tmp, cfg)
            # Local key wins → Iris present, Jake absent
            self.assertIn('Iris', speaker_map)
            self.assertNotIn('Jake', speaker_map)
            # use_unknown_prefix = True because key_path is set
            self.assertTrue(use_prefix)
        finally:
            shutil.rmtree(import_dir, ignore_errors=True)

    # ---- Error tolerance ----

    def test_malformed_json_returns_empty_map(self):
        """A corrupt key file must not raise; empty map returned."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        with open(key_path, 'w') as f:
            f.write('THIS IS NOT JSON {{{{')
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(speaker_map, {})

    def test_empty_json_object_gives_empty_map(self):
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {})
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(speaker_map, {})

    # ---- Key file roundtrip ----

    def test_key_roundtrip_all_speakers_preserved(self):
        """Write multiple speakers to key file; reload and verify all present."""
        entries = {
            'Participant_A': {'role': 'participant', 'anonymized_id': 'P_001'},
            'Participant_B': {'role': 'participant', 'anonymized_id': 'P_002'},
            'Therapist_X':   {'role': 'therapist',   'anonymized_id': 'T_001'},
            'Therapist_Y':   {'role': 'therapist',   'anonymized_id': 'T_002'},
        }
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, entries)
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        self.assertEqual(len(speaker_map), len(entries))
        for name, data in entries.items():
            self.assertIn(name, speaker_map)
            role, anon_id = speaker_map[name]
            self.assertEqual(role, data['role'])
            self.assertEqual(anon_id, data['anonymized_id'])

    def test_key_roundtrip_preserves_anonymized_ids_exactly(self):
        """Anonymized IDs must survive the JSON roundtrip byte-for-byte."""
        special_id = 'PARTICIPANT_Cohort1_Session3_ID0042'
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {
            'Real Name': {'role': 'participant', 'anonymized_id': special_id},
        })
        speaker_map, _ = load_speaker_map(self.tmp, _Cfg())
        _, loaded_id = speaker_map['Real Name']
        self.assertEqual(loaded_id, special_id)

    def test_use_unknown_prefix_false_when_no_config_key_path(self):
        """use_unknown_prefix is False when speaker_anonymization_key_path is None."""
        key_path = os.path.join(self.tmp, 'speaker_anonymization_key.json')
        _write_key(key_path, {'X': {'role': 'participant', 'anonymized_id': 'P_X'}})
        _, use_prefix = load_speaker_map(self.tmp, _Cfg(key_path=None))
        self.assertFalse(use_prefix)


if __name__ == '__main__':
    unittest.main()
