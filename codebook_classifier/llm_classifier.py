"""
llm_classifier.py
-----------------
Multi-label codebook classification using zero-shot LLM prompting.

For each segment, the LLM is prompted with the full codebook and asked
to identify ALL codes that apply (true multi-label output).
"""

import json
from typing import Dict, List, Optional

from shared.data_structures import Segment
from shared.llm_client import LLMClient, extract_json
from shared.classification_loop import filter_participant_segments, classify_segments
from shared.majority_vote import multi_label_majority_vote
from .codebook_schema import Codebook, CodeAssignment
from .config import LLMCodebookConfig


CODEBOOK_PROMPT_TEMPLATE = """You are a qualitative researcher applying a \
codebook to therapeutic dialogue transcripts.

The codebook contains {n_codes} codes organized into {n_domains} domains:

{codebook_string}

For the following participant utterance, identify ALL codes that apply. \
A segment may have zero, one, or multiple codes.

Utterance:
```
{text}
```

Provide your classification as JSON with these exact fields:
{{
    "applied_codes": [
        {{
            "code": "<code category name>",
            "confidence": <float 0-1>,
            "justification": "<brief explanation>"
        }}
    ],
    "no_codes_apply": <boolean>
}}

Rules:
- Apply ALL codes whose inclusion criteria are met by this utterance
- Do NOT apply codes whose exclusion criteria are met
- Confidence should reflect how clearly the inclusion criteria are met (1.0 = textbook match)
- If no codes apply, set no_codes_apply to true and leave applied_codes empty
- Do NOT provide any text outside the JSON

JSON:"""


class LLMCodebookClassifier:
    """Multi-label codebook classification via LLM zero-shot prompting."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[LLMCodebookConfig] = None,
    ):
        self.client = llm_client
        self.config = config or LLMCodebookConfig()

    def classify_segments(
        self,
        segments: List[Segment],
        codebook: Codebook,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[CodeAssignment]]:
        """
        Classify segments using LLM with multi-label codebook prompt.

        Returns dict mapping segment_id -> list of CodeAssignments.
        """
        valid_categories = {c.category.lower(): c for c in codebook.codes}

        participant_segments = filter_participant_segments(segments)

        def build_prompt(segment: Segment, run: int) -> str:
            return self._build_prompt(segment, codebook)

        def parse_response(response: str) -> Optional[List[CodeAssignment]]:
            return self._parse_response(response, valid_categories)

        def merge_runs(parsed_runs: List[List[CodeAssignment]]) -> List[CodeAssignment]:
            return self._merge_runs(parsed_runs)

        def serialize_result(assignments: List[CodeAssignment]):
            return [
                {
                    'code_id': a.code_id,
                    'category': a.category,
                    'confidence': a.confidence,
                    'justification': a.justification,
                }
                for a in assignments
            ]

        effective_output = output_dir or self.config.output_dir or None

        raw_results = classify_segments(
            segments=participant_segments,
            client=self.client,
            n_runs=self.config.n_runs,
            build_prompt=build_prompt,
            parse_response=parse_response,
            merge_runs=merge_runs,
            output_dir=effective_output,
            save_interval=self.config.save_interval,
            file_prefix='codebook_llm_results',
            serialize_result=serialize_result,
        )

        return raw_results

    def _build_prompt(self, segment: Segment, codebook: Codebook) -> str:
        """Build the multi-label classification prompt."""
        codebook_string = codebook.to_prompt_string(
            randomize=self.config.randomize_codebook
        )
        return CODEBOOK_PROMPT_TEMPLATE.format(
            n_codes=len(codebook.codes),
            n_domains=len(codebook.domain_names),
            codebook_string=codebook_string,
            text=segment.text,
        )

    def _parse_response(
        self,
        response: str,
        valid_categories: Dict[str, 'CodeDefinition'],
    ) -> List[CodeAssignment]:
        """Parse a single LLM response into code assignments."""
        try:
            parsed = extract_json(response)
        except (ValueError, json.JSONDecodeError):
            return []

        assignments = []
        for entry in parsed.get('applied_codes', []):
            code_name = str(entry.get('code', '')).strip()
            code_lower = code_name.lower()

            code_def = valid_categories.get(code_lower)
            if code_def is None:
                continue

            confidence = float(entry.get('confidence', 0.5))
            if confidence < self.config.confidence_threshold:
                continue

            assignments.append(CodeAssignment(
                code_id=code_def.code_id,
                category=code_def.category,
                confidence=confidence,
                justification=str(entry.get('justification', '')),
                method='llm',
            ))

        return assignments

    def _merge_runs(
        self, all_assignments: List[List[CodeAssignment]]
    ) -> List[CodeAssignment]:
        """Merge assignments across multiple runs via majority voting."""
        if not all_assignments:
            return []

        voted = multi_label_majority_vote(all_assignments)

        return [
            CodeAssignment(
                code_id=exemplar.code_id,
                category=exemplar.category,
                confidence=round(avg_conf, 4),
                justification=exemplar.justification,
                method='llm',
            )
            for exemplar, avg_conf in voted
        ]
