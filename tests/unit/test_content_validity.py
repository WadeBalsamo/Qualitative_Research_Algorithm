"""Tests for process/assembly/content_validity.py — CV freeze + refresh logic."""
import json
import os
import sys
import tempfile
import time
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process._freeze import FrozenArtifactError
from process import output_paths as _paths
from theme_framework.theme_schema import ThemeDefinition, ThemeFramework


def _make_mini_framework(name='TESTFW', version='1.0', n_themes=2):
    """Create a minimal ThemeFramework with exemplar/subtle/adversarial utterances."""
    themes = []
    for i in range(n_themes):
        themes.append(ThemeDefinition(
            theme_id=i,
            key=f'T{i}',
            name=f'Theme {i}',
            short_name=f'T{i}',
            prompt_name=f'theme_{i}',
            definition=f'Definition of theme {i}.',
            prototypical_features=[f'Feature {i}a', f'Feature {i}b'],
            distinguishing_criteria=f'Distinguishing criterion for theme {i}.',
            exemplar_utterances=[f'Clear example for theme {i}.'],
            subtle_utterances=[f'Subtle example for theme {i}.'],
            adversarial_utterances=[f'Adversarial example for theme {i}.'],
        ))
    return ThemeFramework(name=name, version=version, description='Test framework', themes=themes)


def _make_empty_framework():
    """Framework with no utterances (edge case)."""
    return ThemeFramework(name='EMPTY', version='1.0', description='Empty', themes=[
        ThemeDefinition(
            theme_id=0, key='E', name='Empty', short_name='E', prompt_name='empty',
            definition='Empty def.', prototypical_features=[], distinguishing_criteria='',
            exemplar_utterances=[], subtle_utterances=[], adversarial_utterances=[],
        )
    ])


class TestCreateFrozenContentValidityTestset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_all_artifacts_for_vaamr(self):
        from process.assembly.content_validity import (
            create_frozen_content_validity_testset, read_cv_manifest, read_cv_items,
        )
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        # manifest + items now live in qra.db, not manifest.json / items.jsonl
        self.assertIsNotNone(read_cv_manifest(self.tmpdir, 'cv_vaamr_v1'))
        self.assertTrue(read_cv_items(self.tmpdir, 'cv_vaamr_v1'))
        # the .txt human-facing artifacts stay on disk
        self.assertTrue(os.path.isfile(_paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_vaamr_v1')))
        self.assertTrue(os.path.isfile(_paths.cv_testset_definition_key_path(self.tmpdir, 'cv_vaamr_v1')))
        self.assertTrue(os.path.isfile(_paths.cv_testset_answer_key_path(self.tmpdir, 'cv_vaamr_v1')))

    def test_creates_all_artifacts_for_purer(self):
        from process.assembly.content_validity import (
            create_frozen_content_validity_testset, read_cv_manifest, read_cv_items,
        )
        framework = _make_mini_framework(name='PURER')
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_purer_v1', kind='purer',
        )
        # manifest + items now live in qra.db, not manifest.json / items.jsonl
        self.assertIsNotNone(read_cv_manifest(self.tmpdir, 'cv_purer_v1'))
        self.assertTrue(read_cv_items(self.tmpdir, 'cv_purer_v1'))
        # the .txt human-facing artifacts stay on disk
        self.assertTrue(os.path.isfile(_paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_purer_v1')))
        self.assertTrue(os.path.isfile(_paths.cv_testset_definition_key_path(self.tmpdir, 'cv_purer_v1')))
        self.assertTrue(os.path.isfile(_paths.cv_testset_answer_key_path(self.tmpdir, 'cv_purer_v1')))

    def test_raises_on_second_create_without_force(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        with self.assertRaises(FrozenArtifactError):
            create_frozen_content_validity_testset(
                framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
            )

    def test_force_overwrites_frozen_files(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        ws_path = _paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_vaamr_v1')
        mtime_before = os.path.getmtime(ws_path)
        time.sleep(0.05)
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr', force=True,
        )
        self.assertNotEqual(os.path.getmtime(ws_path), mtime_before)

    def test_manifest_has_required_fields(self):
        from process.assembly.content_validity import (
            create_frozen_content_validity_testset, read_cv_manifest,
        )
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        m = read_cv_manifest(self.tmpdir, 'cv_vaamr_v1')
        self.assertEqual(m['kind'], 'vaamr')
        self.assertIn('framework', m)
        self.assertEqual(m['framework']['name'], 'TESTFW')
        self.assertIn('item_ids', m)
        self.assertIn('content_sha256', m)
        self.assertIn('created_at', m)

    def test_items_jsonl_has_correct_fields(self):
        from process.assembly.content_validity import (
            create_frozen_content_validity_testset, read_cv_items,
        )
        framework = _make_mini_framework(n_themes=2)  # 2 themes × 3 tiers × 1 utterance = 6 items
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        items = read_cv_items(self.tmpdir, 'cv_vaamr_v1')
        self.assertEqual(len(items), 6)  # 2 themes × 3 utterances each
        required_fields = {'id', 'text', 'expected_stage', 'difficulty', 'source_field', 'content_sha256'}
        for item in items:
            self.assertTrue(required_fields.issubset(item.keys()),
                            f"Missing fields: {required_fields - item.keys()}")
        # Difficulties should be clear/subtle/adversarial
        difficulties = {item['difficulty'] for item in items}
        self.assertEqual(difficulties, {'clear', 'subtle', 'adversarial'})
        # source_field should be one of the expected values
        source_fields = {item['source_field'] for item in items}
        self.assertTrue(source_fields.issubset({'exemplar', 'subtle', 'adversarial'}))

    def test_human_worksheet_is_blind(self):
        """Worksheet must not reveal expected stage."""
        from process.assembly.content_validity import create_frozen_content_validity_testset
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        with open(_paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_vaamr_v1')) as f:
            content = f.read()
        # Should not contain "Expected:" or "expected_stage"
        self.assertNotIn('Expected:', content)
        self.assertNotIn('expected_stage', content)
        # Should contain stage labels for reference
        self.assertIn('TESTFW', content)

    def test_definition_key_shows_framework_definitions(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset
        framework = _make_mini_framework()
        create_frozen_content_validity_testset(
            framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )
        with open(_paths.cv_testset_definition_key_path(self.tmpdir, 'cv_vaamr_v1')) as f:
            content = f.read()
        self.assertIn('TESTFW', content)
        self.assertIn('THEME 0', content)
        self.assertIn('THEME 1', content)

    def test_codebook_kind_raises_not_implemented(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset
        framework = _make_mini_framework()
        with self.assertRaises(NotImplementedError):
            create_frozen_content_validity_testset(
                framework, self.tmpdir, name='cv_codebook', kind='codebook',
            )


class TestRefreshCvAnswerKey(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.framework = _make_mini_framework()
        from process.assembly.content_validity import create_frozen_content_validity_testset
        create_frozen_content_validity_testset(
            self.framework, self.tmpdir, name='cv_vaamr_v1', kind='vaamr',
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_refresh_updates_answer_key(self):
        from process.assembly.content_validity import refresh_cv_answer_key
        ai_path = _paths.cv_testset_answer_key_path(self.tmpdir, 'cv_vaamr_v1')
        mtime_before = os.path.getmtime(ai_path)
        time.sleep(0.05)
        refresh_cv_answer_key(self.tmpdir, 'cv_vaamr_v1', None, self.framework)
        self.assertNotEqual(os.path.getmtime(ai_path), mtime_before)

    def test_refresh_does_not_touch_frozen_files(self):
        # The frozen things are now the .txt worksheet/definition key on disk
        # plus the cv_testsets / cv_testset_items DB rows; refresh must only
        # rewrite AI_answer_key.txt.
        from process.assembly.content_validity import refresh_cv_answer_key, read_cv_manifest
        from process import db
        ws_path = _paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_vaamr_v1')
        dk_path = _paths.cv_testset_definition_key_path(self.tmpdir, 'cv_vaamr_v1')
        ai_path = _paths.cv_testset_answer_key_path(self.tmpdir, 'cv_vaamr_v1')
        ws_mtime = os.path.getmtime(ws_path)
        dk_mtime = os.path.getmtime(dk_path)
        ai_mtime = os.path.getmtime(ai_path)
        manifest_before = read_cv_manifest(self.tmpdir, 'cv_vaamr_v1')
        with db.open_db(self.tmpdir) as c:
            created_at_before = c.execute(
                "SELECT created_at FROM cv_testsets WHERE name = ?",
                ('cv_vaamr_v1',),
            ).fetchone()['created_at']
        time.sleep(0.05)
        refresh_cv_answer_key(self.tmpdir, 'cv_vaamr_v1', None, self.framework)
        # Frozen .txt artifacts untouched
        self.assertEqual(os.path.getmtime(ws_path), ws_mtime)
        self.assertEqual(os.path.getmtime(dk_path), dk_mtime)
        # AI answer key rewritten
        self.assertNotEqual(os.path.getmtime(ai_path), ai_mtime)
        # Frozen DB rows (manifest + created_at) untouched
        self.assertEqual(read_cv_manifest(self.tmpdir, 'cv_vaamr_v1'), manifest_before)
        with db.open_db(self.tmpdir) as c:
            created_at_after = c.execute(
                "SELECT created_at FROM cv_testsets WHERE name = ?",
                ('cv_vaamr_v1',),
            ).fetchone()['created_at']
        self.assertEqual(created_at_after, created_at_before)

    def test_refresh_detects_item_drift(self):
        """If an item's text drifts from its stored content_sha256, refresh should
        raise FrozenArtifactError."""
        from process.assembly.content_validity import refresh_cv_answer_key
        from process import db
        # Tamper with the stored item text in the DB — changing text but NOT
        # content_sha256, so the SHA check detects drift.
        with db.open_db(self.tmpdir) as c:
            first_id = c.execute(
                "SELECT item_id FROM cv_testset_items WHERE testset_name = ? "
                "ORDER BY ord LIMIT 1",
                ('cv_vaamr_v1',),
            ).fetchone()['item_id']
            c.execute(
                "UPDATE cv_testset_items SET text = ? "
                "WHERE testset_name = ? AND item_id = ?",
                ('TAMPERED TEXT THAT IS DIFFERENT', 'cv_vaamr_v1', first_id),
            )
        with self.assertRaises(FrozenArtifactError):
            refresh_cv_answer_key(self.tmpdir, 'cv_vaamr_v1', None, self.framework)


class TestListContentValidityTestsets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_returns_empty_when_none(self):
        from process.assembly.content_validity import list_content_validity_testsets
        results = list_content_validity_testsets(self.tmpdir)
        self.assertEqual(results, [])

    def test_list_returns_one_entry_per_testset(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset, list_content_validity_testsets
        fw = _make_mini_framework()
        create_frozen_content_validity_testset(fw, self.tmpdir, name='cv_test_1', kind='vaamr')
        create_frozen_content_validity_testset(fw, self.tmpdir, name='cv_test_2', kind='vaamr')
        results = list_content_validity_testsets(self.tmpdir)
        self.assertEqual(len(results), 2)
        names = {r['name'] for r in results}
        self.assertIn('cv_test_1', names)
        self.assertIn('cv_test_2', names)

    def test_list_entry_has_required_fields(self):
        from process.assembly.content_validity import create_frozen_content_validity_testset, list_content_validity_testsets
        fw = _make_mini_framework()
        create_frozen_content_validity_testset(fw, self.tmpdir, name='cv_test_1', kind='vaamr')
        results = list_content_validity_testsets(self.tmpdir)
        entry = results[0]
        self.assertIn('name', entry)
        self.assertIn('kind', entry)
        self.assertIn('n_items', entry)
        self.assertIn('created_at', entry)
        self.assertEqual(entry['kind'], 'vaamr')
        self.assertGreater(entry['n_items'], 0)


class TestGenerateOrRefreshCVCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_on_first_call(self):
        from process.assembly.content_validity import (
            generate_or_refresh_content_validity_testsets, read_cv_manifest,
        )
        from process.config import ContentValidityConfig, ContentValiditySpec
        fw = _make_mini_framework()
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
            purer=ContentValiditySpec(enabled=False, name='cv_purer_v1'),
        )
        dirs = generate_or_refresh_content_validity_testsets(
            self.tmpdir,
            cv_config=cv_cfg,
            framework_vaamr=fw,
            framework_purer=None,
            theme_classification_cfg=None,
        )
        self.assertEqual(len(dirs), 1)
        self.assertIsNotNone(read_cv_manifest(self.tmpdir, 'cv_vaamr_v1'))

    def test_refreshes_on_second_call(self):
        from process.assembly.content_validity import generate_or_refresh_content_validity_testsets
        from process.config import ContentValidityConfig, ContentValiditySpec
        fw = _make_mini_framework()
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
            purer=ContentValiditySpec(enabled=False, name='cv_purer_v1'),
        )
        generate_or_refresh_content_validity_testsets(
            self.tmpdir, cv_config=cv_cfg, framework_vaamr=fw,
            framework_purer=None, theme_classification_cfg=None,
        )
        ws_path = _paths.cv_testset_human_worksheet_path(self.tmpdir, 'cv_vaamr_v1')
        ws_mtime = os.path.getmtime(ws_path)
        time.sleep(0.05)
        generate_or_refresh_content_validity_testsets(
            self.tmpdir, cv_config=cv_cfg, framework_vaamr=fw,
            framework_purer=None, theme_classification_cfg=None,
        )
        self.assertEqual(os.path.getmtime(ws_path), ws_mtime)

    def test_creates_both_when_both_enabled(self):
        from process.assembly.content_validity import (
            generate_or_refresh_content_validity_testsets, read_cv_manifest,
        )
        from process.config import ContentValidityConfig, ContentValiditySpec
        fw_vaamr = _make_mini_framework(name='VAAMR')
        fw_purer = _make_mini_framework(name='PURER')
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
            purer=ContentValiditySpec(enabled=True, name='cv_purer_v1'),
        )
        dirs = generate_or_refresh_content_validity_testsets(
            self.tmpdir, cv_config=cv_cfg,
            framework_vaamr=fw_vaamr, framework_purer=fw_purer,
            theme_classification_cfg=None,
        )
        self.assertEqual(len(dirs), 2)
        self.assertIsNotNone(read_cv_manifest(self.tmpdir, 'cv_vaamr_v1'))
        self.assertIsNotNone(read_cv_manifest(self.tmpdir, 'cv_purer_v1'))

    def test_any_enabled_false_when_both_disabled(self):
        from process.config import ContentValidityConfig, ContentValiditySpec
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=False, name='cv_vaamr'),
            purer=ContentValiditySpec(enabled=False, name='cv_purer'),
        )
        self.assertFalse(cv_cfg.any_enabled())

    def test_any_enabled_true_when_vaamr_enabled(self):
        from process.config import ContentValidityConfig, ContentValiditySpec
        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_vaamr'),
            purer=ContentValiditySpec(enabled=False, name='cv_purer'),
        )
        self.assertTrue(cv_cfg.any_enabled())


class TestConfigLegacyFallback(unittest.TestCase):
    """Test that PipelineConfig.from_json handles old TestSetConfig format."""

    def test_old_test_sets_format_migrated(self):
        from process.config import PipelineConfig, TestSetsConfig
        old_data = {
            'test_sets': {
                'enabled': True,
                'n_sets': 3,
                'fraction_per_set': 0.15,
                'random_seed': 99,
            }
        }
        cfg = PipelineConfig.from_json(old_data)
        self.assertIsInstance(cfg.test_sets, TestSetsConfig)
        self.assertTrue(cfg.test_sets.vaamr.enabled)
        self.assertEqual(cfg.test_sets.vaamr.n_sets, 3)
        self.assertAlmostEqual(cfg.test_sets.vaamr.fraction_per_set, 0.15)
        self.assertEqual(cfg.test_sets.vaamr.random_seed, 99)

    def test_new_test_sets_format_loaded(self):
        from process.config import PipelineConfig, TestSetsConfig, TestSetSpec
        new_data = {
            'test_sets': {
                'vaamr': {'enabled': True, 'name': 'my_vaamr', 'n_sets': 1,
                          'fraction_per_set': 0.1, 'random_seed': 42},
                'purer': {'enabled': True, 'name': 'my_purer', 'n_sets': 1,
                          'fraction_per_set': 0.1, 'random_seed': 42},
                'codebook': {'enabled': False, 'name': 'cb', 'n_sets': 1,
                             'fraction_per_set': 0.1, 'random_seed': 42},
            }
        }
        cfg = PipelineConfig.from_json(new_data)
        self.assertIsInstance(cfg.test_sets, TestSetsConfig)
        self.assertTrue(cfg.test_sets.vaamr.enabled)
        self.assertEqual(cfg.test_sets.vaamr.name, 'my_vaamr')
        self.assertTrue(cfg.test_sets.purer.enabled)
        self.assertFalse(cfg.test_sets.codebook.enabled)

    def test_content_validity_config_loaded(self):
        from process.config import PipelineConfig, ContentValidityConfig
        data = {
            'content_validity': {
                'vaamr': {'enabled': True, 'name': 'cv_vaamr_v1'},
                'purer': {'enabled': False, 'name': 'cv_purer_v1'},
            }
        }
        cfg = PipelineConfig.from_json(data)
        self.assertIsInstance(cfg.content_validity, ContentValidityConfig)
        self.assertTrue(cfg.content_validity.vaamr.enabled)
        self.assertEqual(cfg.content_validity.vaamr.name, 'cv_vaamr_v1')
        self.assertFalse(cfg.content_validity.purer.enabled)


if __name__ == '__main__':
    unittest.main()
