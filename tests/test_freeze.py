"""Tests for process/_freeze.py."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process._freeze import FrozenArtifactError, sha256_text, verify_content_sha, write_frozen


class TestWriteFrozen(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_file(self):
        path = os.path.join(self.tmpdir, 'out.txt')
        write_frozen(path, lambda fh: fh.write('hello'))
        with open(path) as f:
            self.assertEqual(f.read(), 'hello')

    def test_raises_on_overwrite(self):
        path = os.path.join(self.tmpdir, 'out.txt')
        write_frozen(path, lambda fh: fh.write('first'))
        with self.assertRaises(FrozenArtifactError):
            write_frozen(path, lambda fh: fh.write('second'))
        with open(path) as f:
            self.assertEqual(f.read(), 'first')

    def test_force_overwrites(self):
        path = os.path.join(self.tmpdir, 'out.txt')
        write_frozen(path, lambda fh: fh.write('first'))
        write_frozen(path, lambda fh: fh.write('second'), force=True)
        with open(path) as f:
            self.assertEqual(f.read(), 'second')

    def test_creates_parent_dirs(self):
        path = os.path.join(self.tmpdir, 'a', 'b', 'out.txt')
        write_frozen(path, lambda fh: fh.write('nested'))
        self.assertTrue(os.path.isfile(path))

    def test_cleans_up_tmp_on_error(self):
        path = os.path.join(self.tmpdir, 'out.txt')
        with self.assertRaises(ValueError):
            write_frozen(path, lambda fh: (_ for _ in ()).throw(ValueError('boom')))
        self.assertFalse(os.path.exists(path + '.tmp'))


class TestSha256Text(unittest.TestCase):
    def test_consistent(self):
        h = sha256_text('hello world')
        self.assertEqual(h, sha256_text('hello world'))
        self.assertEqual(len(h), 64)

    def test_differs_on_change(self):
        self.assertNotEqual(sha256_text('abc'), sha256_text('xyz'))


class TestVerifyContentSha(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_snapshot(self, records):
        path = os.path.join(self.tmpdir, 'snapshot.jsonl')
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        return path

    def _mock_seg(self, segment_id, text):
        from unittest.mock import MagicMock
        seg = MagicMock()
        seg.segment_id = segment_id
        seg.text = text
        return seg

    def test_returns_true_for_matching_sha(self):
        text = 'exact match'
        seg = self._mock_seg('seg_1', text)
        snap = self._make_snapshot([{
            'segment_id': 'seg_1',
            'content_sha256': sha256_text(text),
        }])
        results = verify_content_sha(snap, {'seg_1': seg})
        self.assertTrue(results['seg_1'])

    def test_returns_false_for_drifted_text(self):
        seg = self._mock_seg('seg_1', 'changed text')
        snap = self._make_snapshot([{
            'segment_id': 'seg_1',
            'content_sha256': sha256_text('original text'),
        }])
        results = verify_content_sha(snap, {'seg_1': seg})
        self.assertFalse(results['seg_1'])

    def test_returns_false_for_missing_segment(self):
        snap = self._make_snapshot([{
            'segment_id': 'missing_seg',
            'content_sha256': sha256_text('anything'),
        }])
        results = verify_content_sha(snap, {})
        self.assertFalse(results['missing_seg'])


if __name__ == '__main__':
    unittest.main()
