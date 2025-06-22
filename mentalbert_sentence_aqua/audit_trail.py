"""
audit_trail.py
--------------

Implements transparent logging and audit trails for all classification decisions. Tracks:
- Sentence-level and community-level label assignments.
- Confidence scores and rationale for each decision.
- Graph structure and clustering steps.
- Enables full reproducibility and review of automated coding for qualitative researchers.
"""

import json
from typing import Any, Dict, List

class AuditTrail:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log_decision(self, step: str, data: Any) -> None:
        """
        Log a decision or step in the pipeline for transparency and reproducibility.
        Args:
            step: Name of the pipeline step (e.g., 'sentence_labels', 'community_labels', 'evaluation').
            data: Data or rationale to log for this step.
        """
        self.entries.append({'step': step, 'data': data})

    def save(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)

    def get_entries(self, step: str = "") -> List[Dict[str, Any]]:
        if not step:
            return self.entries
        return [e for e in self.entries if e['step'] == step]
