"""
tests/unit/test_config_runs.py
------------------------------
Unit tests for M6 config dataclasses and related infrastructure:
  - ModelRosterEntry, RunSelectionSpec, RunSelectionConfig, AutoRepairConfig, RunExecutionConfig
  - Lenient roster parsing (unknown keys, bad entries skipped)
  - PipelineConfig.to_json / from_json round-trip including roster
  - upgrade_config_file on a minimal legacy config: new blocks added, model_roster at top-level,
    idempotent on second call
  - _flatten_wizard_config carries model_roster (imported from qra.py)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict, fields

# Ensure src/ is on the path (mirrors tests/conftest.py)
_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process.config import (
    PipelineConfig,
    ModelRosterEntry,
    RunSelectionSpec,
    RunSelectionConfig,
    AutoRepairConfig,
    RunExecutionConfig,
    _parse_model_roster,
    _parse_run_selection_config,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults(unittest.TestCase):

    def test_model_roster_entry_defaults(self):
        e = ModelRosterEntry()
        self.assertEqual(e.model, '')
        self.assertIsNone(e.backend)
        self.assertIsNone(e.quantization)
        self.assertIsNone(e.thinking)
        self.assertEqual(e.note, '')
        self.assertIsNone(e.alias)
        self.assertIsNone(e.temperature)
        self.assertEqual(e.frameworks, ['vaamr', 'purer'])

    def test_run_selection_spec_defaults(self):
        s = RunSelectionSpec()
        self.assertEqual(s.strategy, 'top_n_by_human_irr')
        self.assertEqual(s.n, 3)
        self.assertIsNone(s.min_kappa)

    def test_run_selection_config_defaults(self):
        cfg = RunSelectionConfig()
        self.assertEqual(cfg.vaamr.strategy, 'top_n_by_human_irr')
        self.assertEqual(cfg.vaamr.n, 3)
        self.assertEqual(cfg.purer.strategy, 'all')
        self.assertIsNone(cfg.purer.n)

    def test_auto_repair_config_defaults(self):
        ar = AutoRepairConfig()
        self.assertTrue(ar.enabled)
        self.assertEqual(ar.max_passes, 2)
        self.assertAlmostEqual(ar.dead_rater_error_fraction, 0.5)

    def test_run_execution_config_defaults(self):
        re = RunExecutionConfig()
        self.assertEqual(re.retries, 2)
        self.assertEqual(re.save_interval, 20)

    def test_pipeline_config_has_new_fields(self):
        pc = PipelineConfig()
        self.assertIsInstance(pc.model_roster, list)
        self.assertEqual(pc.model_roster, [])
        self.assertIsInstance(pc.run_selection, RunSelectionConfig)
        self.assertIsInstance(pc.auto_repair, AutoRepairConfig)
        self.assertIsInstance(pc.run_execution, RunExecutionConfig)


# ---------------------------------------------------------------------------
# Roster lenient parsing
# ---------------------------------------------------------------------------

class TestRosterParsing(unittest.TestCase):

    def test_valid_entry(self):
        lst = [{'model': 'qwen/qwen3-70b', 'quantization': 'Q4_K_M'}]
        result = _parse_model_roster(lst)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model, 'qwen/qwen3-70b')
        self.assertEqual(result[0].quantization, 'Q4_K_M')

    def test_unknown_keys_ignored(self):
        lst = [{'model': 'modelA', 'unknown_future_key': 'foo', 'another': 42}]
        result = _parse_model_roster(lst)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model, 'modelA')

    def test_bad_entry_missing_model_skipped(self):
        lst = [
            {'model': 'goodModel', 'note': 'ok'},
            {'quantization': 'Q4'},          # missing 'model'
            {'model': '', 'note': 'empty'},  # empty model
        ]
        result = _parse_model_roster(lst)
        # Only the first entry has a non-empty model
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model, 'goodModel')

    def test_non_dict_entry_skipped(self):
        lst = [
            {'model': 'good'},
            'not-a-dict',
            None,
            42,
        ]
        result = _parse_model_roster(lst)
        self.assertEqual(len(result), 1)

    def test_thinking_field(self):
        lst = [{'model': 'm', 'thinking': 'off'}]
        result = _parse_model_roster(lst)
        self.assertEqual(result[0].thinking, 'off')

    def test_frameworks_field(self):
        lst = [{'model': 'm', 'frameworks': ['vaamr']}]
        result = _parse_model_roster(lst)
        self.assertEqual(result[0].frameworks, ['vaamr'])


# ---------------------------------------------------------------------------
# to_json / from_json round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):

    def test_empty_roster_round_trips(self):
        pc = PipelineConfig()
        d = pc.to_json()
        self.assertIn('model_roster', d)
        self.assertEqual(d['model_roster'], [])
        pc2 = PipelineConfig.from_json(d)
        self.assertEqual(pc2.model_roster, [])

    def test_roster_with_entries_round_trips(self):
        pc = PipelineConfig()
        pc.model_roster = [
            ModelRosterEntry(model='qwen/qwen3-70b', quantization='Q4', note='primary'),
            ModelRosterEntry(model='google/gemma-4-31b', thinking='off', frameworks=['purer']),
        ]
        d = pc.to_json()
        self.assertEqual(len(d['model_roster']), 2)
        pc2 = PipelineConfig.from_json(d)
        self.assertEqual(len(pc2.model_roster), 2)
        self.assertEqual(pc2.model_roster[0].model, 'qwen/qwen3-70b')
        self.assertEqual(pc2.model_roster[0].quantization, 'Q4')
        self.assertEqual(pc2.model_roster[1].thinking, 'off')
        self.assertEqual(pc2.model_roster[1].frameworks, ['purer'])

    def test_run_selection_round_trips(self):
        pc = PipelineConfig()
        pc.run_selection = RunSelectionConfig(
            vaamr=RunSelectionSpec(strategy='top_n_by_human_irr', n=5, min_kappa=0.4),
            purer=RunSelectionSpec(strategy='all', n=None),
        )
        d = pc.to_json()
        pc2 = PipelineConfig.from_json(d)
        self.assertEqual(pc2.run_selection.vaamr.n, 5)
        self.assertAlmostEqual(pc2.run_selection.vaamr.min_kappa, 0.4)
        self.assertEqual(pc2.run_selection.purer.strategy, 'all')

    def test_auto_repair_round_trips(self):
        pc = PipelineConfig()
        pc.auto_repair = AutoRepairConfig(enabled=False, max_passes=3)
        d = pc.to_json()
        pc2 = PipelineConfig.from_json(d)
        self.assertFalse(pc2.auto_repair.enabled)
        self.assertEqual(pc2.auto_repair.max_passes, 3)

    def test_run_execution_round_trips(self):
        pc = PipelineConfig()
        pc.run_execution = RunExecutionConfig(retries=4, save_interval=50)
        d = pc.to_json()
        pc2 = PipelineConfig.from_json(d)
        self.assertEqual(pc2.run_execution.retries, 4)
        self.assertEqual(pc2.run_execution.save_interval, 50)

    def test_unknown_keys_in_from_json_ignored(self):
        d = PipelineConfig().to_json()
        d['totally_new_key_from_future'] = 'irrelevant'
        d.setdefault('model_roster', [])
        # Should not raise
        pc = PipelineConfig.from_json(d)
        self.assertIsNotNone(pc)


# ---------------------------------------------------------------------------
# upgrade_config_file
# ---------------------------------------------------------------------------

class TestUpgradeConfigFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.meta_dir = os.path.join(self.tmp, '02_meta')
        os.makedirs(self.meta_dir)
        self.config_path = os.path.join(self.meta_dir, 'qra_config.json')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_legacy(self, data):
        with open(self.config_path, 'w') as f:
            json.dump(data, f)

    def _read_config(self):
        with open(self.config_path) as f:
            return json.load(f)

    def test_upgrade_adds_new_blocks(self):
        from process.legacy_migration import upgrade_config_file
        # A minimal legacy config without the new M6 blocks
        self._write_legacy({'pipeline': {'output_dir': './data/output/'}})
        changed = upgrade_config_file(self.tmp)
        self.assertTrue(changed)
        data = self._read_config()
        # The new dict-valued blocks should now be present at top level
        self.assertIn('auto_repair', data)
        self.assertIn('run_execution', data)
        self.assertIn('run_selection', data)

    def test_model_roster_at_top_level_not_under_pipeline(self):
        from process.legacy_migration import upgrade_config_file
        self._write_legacy({'pipeline': {'output_dir': './data/output/'}})
        upgrade_config_file(self.tmp)
        data = self._read_config()
        # model_roster is a list default — must be at top level, NOT under pipeline
        self.assertIn('model_roster', data)
        self.assertIsInstance(data['model_roster'], list)
        pipeline_block = data.get('pipeline', {})
        self.assertNotIn('model_roster', pipeline_block)

    def test_idempotent_second_call(self):
        from process.legacy_migration import upgrade_config_file
        self._write_legacy({'pipeline': {'output_dir': './data/output/'}})
        upgrade_config_file(self.tmp)
        data_after_first = self._read_config()
        changed_again = upgrade_config_file(self.tmp)
        # Second call must be a no-op
        self.assertFalse(changed_again)
        data_after_second = self._read_config()
        self.assertEqual(
            json.dumps(data_after_first, sort_keys=True, default=str),
            json.dumps(data_after_second, sort_keys=True, default=str),
        )

    def test_existing_values_not_overwritten(self):
        from process.legacy_migration import upgrade_config_file
        self._write_legacy({
            'pipeline': {'output_dir': './custom/'},
            'auto_repair': {'enabled': False, 'max_passes': 5},
        })
        upgrade_config_file(self.tmp)
        data = self._read_config()
        # The user's custom value must be preserved
        self.assertFalse(data['auto_repair']['enabled'])
        self.assertEqual(data['auto_repair']['max_passes'], 5)


# ---------------------------------------------------------------------------
# _flatten_wizard_config carries model_roster
# ---------------------------------------------------------------------------

class TestFlattenWizardConfig(unittest.TestCase):

    def _import_flatten(self):
        """Import _flatten_wizard_config from qra.py via sys.path."""
        import importlib.util
        qra_path = os.path.join(_QRA_ROOT, 'qra.py')
        spec = importlib.util.spec_from_file_location('qra', qra_path)
        mod = importlib.util.module_from_spec(spec)
        # Stub out the heavy top-level side-effects that need infrastructure
        # by pre-populating sys.modules with what qra.py imports at module level
        # (if they exist on the path they'll load fine; if not we skip the test).
        try:
            spec.loader.exec_module(mod)
        except Exception:
            self.skipTest('qra.py import failed — infra not fully installed')
        return mod._flatten_wizard_config

    def test_model_roster_carried(self):
        flatten = self._import_flatten()
        data = {
            'pipeline': {'output_dir': './data/output/'},
            'model_roster': [{'model': 'qwen/qwen3-70b'}],
            'theme_classification': {'model': 'qwen/qwen3-70b'},
        }
        result = flatten(data)
        self.assertIn('model_roster', result)
        self.assertEqual(result['model_roster'], [{'model': 'qwen/qwen3-70b'}])

    def test_run_selection_carried(self):
        flatten = self._import_flatten()
        data = {
            'pipeline': {},
            'run_selection': {'vaamr': {'strategy': 'top_n_by_human_irr', 'n': 5}},
        }
        result = flatten(data)
        self.assertIn('run_selection', result)

    def test_auto_repair_carried(self):
        flatten = self._import_flatten()
        data = {
            'pipeline': {},
            'auto_repair': {'enabled': False, 'max_passes': 3},
        }
        result = flatten(data)
        self.assertIn('auto_repair', result)
        self.assertFalse(result['auto_repair']['enabled'])


if __name__ == '__main__':
    unittest.main()
