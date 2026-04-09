"""
process_logger.py
-----------------
Simple file-based process logger for segmentation debugging.

Written to process_log.txt in the output directory when
verbose_segmentation=True. Logs every step of the segmentation
pipeline including LLM prompts, responses, and decisions.
"""

import datetime
import os
import textwrap
from typing import Optional


class ProcessLogger:
    """Logs segmentation steps and LLM I/O to a text file."""

    def __init__(self, log_path: Optional[str] = None):
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._fh = open(log_path, 'w', encoding='utf-8')
            self._write(f"QRA Segmentation Process Log")
            self._write(f"Started: {datetime.datetime.utcnow().isoformat()}Z")
            self._write("=" * 80)

    def _write(self, text: str = ''):
        if self._fh:
            print(text, file=self._fh, flush=True)

    def close(self):
        if self._fh:
            self._write()
            self._write(f"Finished: {datetime.datetime.utcnow().isoformat()}Z")
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------
    # Section headers
    # ------------------------------------------------------------------

    def section(self, title: str):
        self._write()
        self._write("=" * 80)
        self._write(f"  {title}")
        self._write("=" * 80)

    def subsection(self, title: str):
        self._write()
        self._write("-" * 60)
        self._write(f"  {title}")
        self._write("-" * 60)

    # ------------------------------------------------------------------
    # Sentence-level logging
    # ------------------------------------------------------------------

    def log_sentences(self, label: str, sentences: list):
        """Log a list of sentence dicts."""
        self._write()
        self._write(f"[{label}] ({len(sentences)} sentences)")
        for i, s in enumerate(sentences):
            spk = s.get('speaker', '?')
            text = s.get('text', '')
            start = s.get('start', 0)
            end = s.get('end', 0)
            self._write(f"  [{i:3d}] {start:8.2f}s-{end:.2f}s  [{spk}]: {text[:120]}")

    def log_sentence_filter(self, before: list, after: list, excluded: set):
        self._write()
        self._write(f"[SPEAKER FILTER] excluded={sorted(excluded) if excluded else 'none'}")
        self._write(f"  Before: {len(before)} sentences  →  After: {len(after)} sentences  "
                    f"(removed {len(before) - len(after)})")

    def log_fold_short(self, before: list, after: list):
        removed = len(before) - len(after)
        if removed:
            self._write(f"[FOLD SHORT SENTENCES]  {len(before)} → {len(after)} "
                        f"(folded {removed} short sentences into neighbors)")

    # ------------------------------------------------------------------
    # Boundary / segmentation logging
    # ------------------------------------------------------------------

    def log_boundaries(self, boundaries: list, boundary_confidence: dict, sentences: list):
        self._write()
        self._write(f"[BOUNDARIES DETECTED]  {len(boundaries)} boundaries")
        for b in boundaries:
            conf = boundary_confidence.get(b, 'n/a') if boundary_confidence else 'n/a'
            spk_before = sentences[b].get('speaker', '?') if b < len(sentences) else '?'
            spk_after = sentences[b + 1].get('speaker', '?') if b + 1 < len(sentences) else '?'
            self._write(f"  idx={b:3d}  conf={conf:10s}  [{spk_before}] → [{spk_after}]")

    def log_segments(self, label: str, segments: list):
        self._write()
        self._write(f"[{label}]  {len(segments)} segments")
        for seg in segments:
            preview = seg.text[:30000].replace('\n', '\\n')
            self._write(f"  {seg.segment_id}  speaker={seg.speaker}  "
                        f"pid={seg.participant_id}  words={seg.word_count}")
            self._write(f"    {preview}")

    def log_merge(self, action: str, seg_a_id: str, seg_b_id: str, reason: str):
        self._write(f"  [MERGE {action}]  {seg_a_id} ← {seg_b_id}  ({reason})")

    # ------------------------------------------------------------------
    # LLM I/O logging
    # ------------------------------------------------------------------

    def log_llm_call(self, call_type: str, prompt: str, response: str, decision: str = ''):
        """Log a full LLM prompt and response."""
        self._write()
        self._write(f"┌─ LLM CALL: {call_type} {'─' * (60 - len(call_type))}")
        self._write("│ PROMPT:")
        for line in textwrap.wrap(prompt, width=76, initial_indent='│   ', subsequent_indent='│   '):
            self._write(line)
        self._write("│")
        self._write("│ RESPONSE:")
        resp_preview = response[:800] if response else '(empty)'
        for line in resp_preview.splitlines():
            self._write(f"│   {line}")
        if decision:
            self._write("│")
            self._write(f"│ DECISION: {decision}")
        self._write("└" + "─" * 70)

    # ------------------------------------------------------------------
    # Context expansion logging
    # ------------------------------------------------------------------

    def log_context_gate(self, seg_id: str, direction: str, sim: float,
                         decision: str, n_sents: int):
        self._write(f"  [CONTEXT {direction.upper():6s}]  seg={seg_id}  "
                    f"sim={sim:.3f}  n_sents={n_sents}  → {decision}")
