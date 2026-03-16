"""
human_validator.py
------------------
Interactive CLI for human validation of uncertain classifications.

Extracted from classify_and_label.py. Validates:
- Theme classifications (low confidence / inconsistency)
- Codebook disagreements (embedding vs LLM mismatch)
- Co-occurrence anomalies (theme-codebook mapping violations)
"""

import datetime
from typing import Dict, List, Optional, Tuple

from classification_tools.data_structures import Segment
from constructs.theme_schema import ThemeFramework
from codebook.ensemble import EnsembleResult


class HumanValidator:
    """Interactive CLI for human validation of uncertain classifications."""

    def __init__(self, framework: ThemeFramework, codebook, skip_confirmation: bool = False):
        self.framework = framework
        self.codebook = codebook
        self.skip_confirmation = skip_confirmation
        self.validation_log: List[Dict] = []

    def validate_theme_classification(
        self,
        segment: Segment,
        reason: str,
    ) -> Optional[Tuple[int, Optional[int]]]:
        """
        Prompt user to validate/correct theme classification.

        Parameters
        ----------
        segment : Segment
            Segment with existing theme labels
        reason : str
            Why this segment needs validation (e.g., "low_confidence", "inconsistency")

        Returns
        -------
        tuple or None
            (primary_stage_id, optional_secondary_stage_id) or None to keep original
        """
        if self.skip_confirmation:
            return None

        self._print_segment_context(segment)
        print(f"\n  Validation Needed: {reason}")
        print(f"\nCurrent Classification:")
        print(f"  Primary: {self.framework.get_theme_by_id(segment.primary_stage).short_name if segment.primary_stage is not None else 'None'}")
        if segment.llm_confidence_primary:
            print(f"  Confidence: {segment.llm_confidence_primary:.2f}")
        if segment.llm_run_consistency:
            print(f"  Consistency: {segment.llm_run_consistency}/3 runs")

        print("\nAvailable Themes:")
        for theme in self.framework.themes:
            print(f"  {theme.theme_id}: {theme.short_name} - {theme.definition[:60]}...")

        response = input("\nEnter primary theme ID (or 'skip' to keep current): ").strip()
        if response.lower() == 'skip':
            return None

        try:
            primary_id = int(response)
            if not 0 <= primary_id < self.framework.num_themes:
                print("Invalid theme ID")
                return None

            secondary_response = input("Enter secondary theme ID (or 'none'): ").strip()
            secondary_id = None
            if secondary_response.lower() != 'none' and secondary_response:
                secondary_id = int(secondary_response)
                if not 0 <= secondary_id < self.framework.num_themes:
                    print("Invalid secondary theme ID")
                    return None

            self.validation_log.append({
                'segment_id': segment.segment_id,
                'type': 'theme',
                'reason': reason,
                'original_primary': segment.primary_stage,
                'new_primary': primary_id,
                'new_secondary': secondary_id,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            return (primary_id, secondary_id)
        except ValueError:
            print("Invalid input")
            return None

    def validate_codebook_disagreement(
        self,
        segment: Segment,
        ensemble_result: EnsembleResult,
    ) -> Optional[List[str]]:
        """
        Prompt user to reconcile embedding vs LLM codebook disagreement.

        Parameters
        ----------
        segment : Segment
            The segment
        ensemble_result : EnsembleResult
            Codebook ensemble results with disagreements

        Returns
        -------
        list of str or None
            Corrected list of code IDs, or None to keep ensemble result
        """
        if self.skip_confirmation:
            return None

        self._print_segment_context(segment)
        print(f"\n  Codebook Disagreement Detected")
        print(f"\nEmbedding-based codes: {', '.join(ensemble_result.embedding_only_codes) or 'none'}")
        print(f"LLM-based codes: {', '.join(ensemble_result.llm_only_codes) or 'none'}")
        print(f"Agreed codes: {', '.join(ensemble_result.agreed_codes) or 'none'}")

        print("\nAvailable codes (domain | category):")
        for domain in self.codebook.domain_names:
            codes = self.codebook.get_codes_by_domain(domain)
            print(f"\n  {domain}:")
            for code in codes:
                print(f"    - {code.code_id}: {code.category}")

        response = input("\nEnter final code IDs (comma-separated) or 'skip': ").strip()
        if response.lower() == 'skip':
            return None

        try:
            final_codes = [c.strip() for c in response.split(',') if c.strip()]
            self.validation_log.append({
                'segment_id': segment.segment_id,
                'type': 'codebook',
                'reason': 'method_disagreement',
                'embedding_codes': ensemble_result.embedding_only_codes,
                'llm_codes': ensemble_result.llm_only_codes,
                'final_codes': final_codes,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            return final_codes
        except Exception as e:
            print(f"Error parsing input: {e}")
            return None

    def validate_cooccurrence_anomaly(
        self,
        segment: Segment,
        anomaly_details: Dict,
    ) -> bool:
        """
        Prompt user to review theme <-> codebook co-occurrence anomaly.

        Parameters
        ----------
        segment : Segment
            The segment
        anomaly_details : Dict
            Information about the anomaly

        Returns
        -------
        bool
            True to accept, False to reject classification
        """
        if self.skip_confirmation:
            return True

        self._print_segment_context(segment)
        print(f"\n  Theme <-> Codebook Anomaly Detected")
        print(f"\nTheme: {self.framework.get_theme_by_id(segment.primary_stage).short_name}")
        print(f"Unexpected code: {anomaly_details.get('code_id')}")
        print(f"Reason: {anomaly_details.get('reason')}")

        response = input("Accept this classification? (y/n): ").strip().lower()
        accept = response == 'y'

        self.validation_log.append({
            'segment_id': segment.segment_id,
            'type': 'cooccurrence',
            'anomaly': anomaly_details,
            'accepted': accept,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        return accept

    def _print_segment_context(self, segment: Segment):
        """Print segment context for human review."""
        print("\n" + "=" * 70)
        print(f"Segment: {segment.segment_id} | Session: {segment.session_id}")
        print(f"Speaker: {segment.speaker} | Words: {segment.word_count}")
        print(f"Time: {segment.start_time_ms / 1000:.1f}s - {segment.end_time_ms / 1000:.1f}s")
        print("-" * 70)
        print(f"Text: {segment.text}")
        print("=" * 70)
