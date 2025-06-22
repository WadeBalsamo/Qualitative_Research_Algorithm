import os
import pytest
from mentalbert_sentence_aqua import preprocessing

def test_parse_srt_file_basic():
    srt_path = os.path.join(os.path.dirname(__file__), '../data/30.srt')
    sentences = preprocessing.parse_srt_file(srt_path)
    assert len(sentences) > 0
    assert hasattr(sentences[0], 'text')
    assert hasattr(sentences[0], 'speaker')
    assert hasattr(sentences[0], 'start')
    assert hasattr(sentences[0], 'end')
    assert hasattr(sentences[0], 'label')
    # Check label extraction
    labels = [s.label for s in sentences if s.label]
    assert any(l for l in labels)

def test_normalize_text():
    raw = 'This is\n a test.  '
    norm = preprocessing.normalize_text(raw)
    assert norm == 'This is a test.'
