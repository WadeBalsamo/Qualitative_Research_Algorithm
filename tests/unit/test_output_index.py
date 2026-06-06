"""
tests/unit/test_output_index.py
--------------------------------
Unit tests for process/output_index.py.

Builds a small fake output tree in a temp directory, calls write_index(),
and verifies 00_index.txt is written and references the expected files.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process.output_index import write_index


def _touch(path: str, content: str = 'x') -> None:
    """Create a file and any missing parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


class TestWriteIndexReturnsPath(unittest.TestCase):
    """write_index() returns the path to 00_index.txt."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_correct_path(self):
        result = write_index(self.tmp)
        expected = os.path.join(self.tmp, '00_index.txt')
        self.assertEqual(result, expected)

    def test_index_file_exists_after_call(self):
        write_index(self.tmp)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, '00_index.txt')))


class TestWriteIndexContent(unittest.TestCase):
    """00_index.txt references the files that were placed in the tree."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Build a minimal but realistic fake output tree
        _touch(os.path.join(self.tmp, '06_reports', '01_outcomes', 'efficacy.txt'),
               'efficacy content')
        _touch(os.path.join(self.tmp, '06_reports', '02_mechanism', 'transitions.txt'),
               'transitions content')
        _touch(os.path.join(self.tmp, '03_analysis_data', 'session_stats', 'ses_001.json'),
               '{}')
        _touch(os.path.join(self.tmp, '05_figures', 'trajectory.png'),
               'PNG')
        _touch(os.path.join(self.tmp, '04_validation', 'testsets',
                            'human_classification_testset_worksheet_1.txt'),
               'sheet')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_index(self) -> str:
        write_index(self.tmp)
        with open(os.path.join(self.tmp, '00_index.txt'), encoding='utf-8') as f:
            return f.read()

    def test_header_contains_run_dir_basename(self):
        content = self._read_index()
        self.assertIn(os.path.basename(self.tmp), content)

    def test_report_file_referenced(self):
        content = self._read_index()
        self.assertIn('efficacy.txt', content)

    def test_mechanism_file_referenced(self):
        content = self._read_index()
        self.assertIn('transitions.txt', content)

    def test_session_stats_file_referenced(self):
        content = self._read_index()
        self.assertIn('ses_001.json', content)

    def test_figure_file_referenced(self):
        content = self._read_index()
        self.assertIn('trajectory.png', content)

    def test_testset_worksheet_referenced(self):
        content = self._read_index()
        self.assertIn('human_classification_testset_worksheet_1.txt', content)

    def test_index_itself_not_listed(self):
        """00_index.txt must not list itself (it is in _SKIP_NAMES)."""
        content = self._read_index()
        # Remove first header line which contains the basename
        body = '\n'.join(content.splitlines()[2:])
        self.assertNotIn('00_index.txt', body)


class TestWriteIndexSkippedDirs(unittest.TestCase):
    """Files inside skipped directories are excluded from the index."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # A normal file that should appear
        _touch(os.path.join(self.tmp, '06_reports', 'visible.txt'), 'ok')
        # Files inside skipped dirs — should NOT appear
        _touch(os.path.join(self.tmp, '02_meta', 'auditable_logs', 'llm_prompts.txt'), 'secret')
        _touch(os.path.join(self.tmp, '02_meta', 'codebook_raw', 'embeddings.npz'), 'bin')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_index(self) -> str:
        write_index(self.tmp)
        with open(os.path.join(self.tmp, '00_index.txt'), encoding='utf-8') as f:
            return f.read()

    def test_visible_file_appears(self):
        self.assertIn('visible.txt', self._read_index())

    def test_auditable_logs_excluded(self):
        self.assertNotIn('llm_prompts.txt', self._read_index())

    def test_codebook_raw_excluded(self):
        self.assertNotIn('embeddings.npz', self._read_index())


class TestWriteIndexFileSizes(unittest.TestCase):
    """File sizes are included in the index output."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _touch(os.path.join(self.tmp, '06_reports', 'small.txt'), 'hello')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_size_marker_present(self):
        """Each entry should include a size unit (B, KB, or MB)."""
        write_index(self.tmp)
        with open(os.path.join(self.tmp, '00_index.txt'), encoding='utf-8') as f:
            content = f.read()
        # small.txt is 5 bytes — should appear as "5 B"
        self.assertIn('5 B', content)


class TestWriteIndexEmptyDir(unittest.TestCase):
    """write_index() on an empty directory still produces a valid file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_produces_file(self):
        write_index(self.tmp)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, '00_index.txt')))

    def test_header_present(self):
        write_index(self.tmp)
        with open(os.path.join(self.tmp, '00_index.txt'), encoding='utf-8') as f:
            first_line = f.readline()
        self.assertIn('QRA Output Index', first_line)


class TestWriteIndexOverwrite(unittest.TestCase):
    """Calling write_index() twice overwrites the old index cleanly."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_call_overwrites(self):
        write_index(self.tmp)
        # Add a new file and regenerate
        _touch(os.path.join(self.tmp, '06_reports', 'new.txt'), 'new content')
        write_index(self.tmp)
        with open(os.path.join(self.tmp, '00_index.txt'), encoding='utf-8') as f:
            content = f.read()
        self.assertIn('new.txt', content)


if __name__ == '__main__':
    unittest.main()
