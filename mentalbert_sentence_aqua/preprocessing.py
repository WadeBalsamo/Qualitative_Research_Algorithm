"""
preprocessing.py
----------------

Handles all data ingestion and preprocessing for the AQUA pipeline. Includes:
- Parsing .srt transcript files into sentence-level units with speaker attribution and timestamps.
- Therapeutic discourse-aware sentence segmentation, preserving utterance boundaries and speaker turns.
- Text normalization and cleaning tailored for therapeutic dialogue (e.g., handling incomplete sentences, overlapping speech).
- Output: List of sentence objects with metadata for downstream embedding and graph construction.
"""

import re
from typing import List, Dict, Any, Optional

class SentenceUnit:
    def __init__(self, text: str, speaker: str, start: str, end: str, idx: int, label: str = "") -> None:
        self.text: str = text
        self.speaker: str = speaker
        self.start: str = start
        self.end: str = end
        self.idx: int = idx
        self.label: str = label if label is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'speaker': self.speaker,
            'start': self.start,
            'end': self.end,
            'idx': self.idx,
            'label': self.label
        }

def parse_srt_file(filepath: str) -> List[SentenceUnit]:
    """
    Parses an .srt transcript file into a list of SentenceUnit objects.
    Assumes each block is a single utterance, with optional speaker and label.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content)
    sentences = []
    idx = 0
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        # Parse timing
        timing = lines[1]
        start, end = timing.split(' --> ')
        # Parse speaker and text
        text_lines = lines[2:]
        speaker = "Unknown"
        text = " ".join(text_lines)
        # Try to extract speaker
        m = re.match(r'(Speaker \d+|\w+):\s*(.*)', text)
        if m:
            speaker = m.group(1)
            text = m.group(2)
        # Try to extract label (e.g., #Reappraisal)
        label = None
        label_match = re.search(r'#(\w+)', text)
        if label_match:
            label = label_match.group(1)
            text = text.replace(f'#{label}', '').strip()
        sentences.append(SentenceUnit(text=text.strip(), speaker=speaker, start=start, end=end, idx=idx, label=label if label is not None else ""))
        idx += 1
    return sentences

def normalize_text(text: str) -> str:
    """
    Basic text normalization for therapeutic dialogue.
    """
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def segment_sentences(units: List[SentenceUnit]) -> List[SentenceUnit]:
    """
    Optionally further segments utterances into sentences, preserving speaker and timing.
    For now, returns input as-is (since .srt is already segmented by utterance).
    """
    # Could use nltk or spacy for more granular segmentation if needed
    return units
