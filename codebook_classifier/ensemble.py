"""
ensemble.py
-----------
Disagreement flagging and reconciliation between embedding-based
and LLM-based codebook classification results.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .codebook_schema import CodeAssignment
from .config import EnsembleConfig


@dataclass
class EnsembleResult:
    """Reconciled codebook classification result for a single segment."""
    segment_id: str
    agreed_codes: List[str] = field(default_factory=list)
    embedding_only_codes: List[str] = field(default_factory=list)
    llm_only_codes: List[str] = field(default_factory=list)
    final_codes: List[str] = field(default_factory=list)
    final_assignments: List[CodeAssignment] = field(default_factory=list)
    needs_human_review: bool = False
    disagreement_details: List[Dict] = field(default_factory=list)


class CodebookEnsemble:
    """Reconciles embedding and LLM codebook classification results."""

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()

    def reconcile(
        self,
        embedding_results: Dict[str, List[CodeAssignment]],
        llm_results: Dict[str, List[CodeAssignment]],
    ) -> Dict[str, EnsembleResult]:
        """
        Reconcile embedding and LLM codebook results per segment.

        For each segment computes the set intersection (agreed),
        symmetric difference (disagreed), and produces a final
        code list based on the configured preferred_method.
        """
        all_segment_ids = set(embedding_results.keys()) | set(llm_results.keys())
        results: Dict[str, EnsembleResult] = {}

        for seg_id in all_segment_ids:
            emb_assignments = embedding_results.get(seg_id, [])
            llm_assignments = llm_results.get(seg_id, [])

            emb_codes = {a.code_id for a in emb_assignments}
            llm_codes = {a.code_id for a in llm_assignments}

            agreed = emb_codes & llm_codes
            emb_only = emb_codes - llm_codes
            llm_only = llm_codes - emb_codes

            # Build lookup by code_id
            emb_by_id = {a.code_id: a for a in emb_assignments}
            llm_by_id = {a.code_id: a for a in llm_assignments}

            # Determine final codes based on config
            if self.config.require_agreement:
                final_code_ids = agreed
            elif self.config.preferred_method == 'llm':
                final_code_ids = llm_codes
            elif self.config.preferred_method == 'embedding':
                final_code_ids = emb_codes
            else:  # 'both' — union
                final_code_ids = emb_codes | llm_codes

            # Build final assignments: prefer LLM metadata, fall back to embedding
            final_assignments = []
            for code_id in sorted(final_code_ids):
                if code_id in llm_by_id:
                    a = llm_by_id[code_id]
                    final_assignments.append(CodeAssignment(
                        code_id=a.code_id,
                        category=a.category,
                        confidence=a.confidence,
                        justification=a.justification,
                        method='ensemble',
                    ))
                elif code_id in emb_by_id:
                    a = emb_by_id[code_id]
                    final_assignments.append(CodeAssignment(
                        code_id=a.code_id,
                        category=a.category,
                        confidence=a.confidence,
                        justification=a.justification,
                        method='ensemble',
                    ))

            # Build disagreement details
            disagreement_details = []
            for code_id in sorted(emb_only):
                a = emb_by_id[code_id]
                disagreement_details.append({
                    'code_id': code_id,
                    'category': a.category,
                    'type': 'embedding_only',
                    'embedding_confidence': a.confidence,
                })
            for code_id in sorted(llm_only):
                a = llm_by_id[code_id]
                disagreement_details.append({
                    'code_id': code_id,
                    'category': a.category,
                    'type': 'llm_only',
                    'llm_confidence': a.confidence,
                })

            has_disagreements = bool(emb_only or llm_only)

            results[seg_id] = EnsembleResult(
                segment_id=seg_id,
                agreed_codes=sorted(agreed),
                embedding_only_codes=sorted(emb_only),
                llm_only_codes=sorted(llm_only),
                final_codes=sorted(final_code_ids),
                final_assignments=final_assignments,
                needs_human_review=(
                    has_disagreements and self.config.flag_disagreements
                ),
                disagreement_details=disagreement_details,
            )

        return results
