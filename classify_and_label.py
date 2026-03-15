#!/usr/bin/env python3
"""
classify_and_label.py
---------------------
Integrated CLI for classification and labeling with human validation.

This module orchestrates the complete pipeline:
1. Transcript ingestion and segmentation
2. Theme/stage classification (zero-shot LLM)
3. Codebook classification (embedding + LLM ensemble)
4. Cross-validation between theme and codebook labels
5. Interactive human validation of uncertain results
6. Final dataset assembly and reporting

The pipeline processes segments and prompts for human validation when:
- Theme classification has low confidence (<0.6) or inconsistency (<3/3 runs)
- Codebook methods disagree (embedding vs LLM)
- Theme ↔ codebook mapping is violated (co-occurrence anomalies)

Usage:
    python classify_and_label.py \\
        --transcript-dir ./data/diarized/ \\
        --output-dir ./data/output/ \\
        --model openai/gpt-4o \\
        --run-mode interactive

Modes:
    auto   : Run fully automated (no human intervention)
    interactive : Prompt for human validation of uncertain results
    review : Review all results at the end (batch validation)
"""

import argparse
import json
import os
import sys
import datetime
from collections import Counter
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import pandas as pd

# Core imports
from shared.data_structures import Segment
from shared.llm_client import LLMClient, LLMClientConfig
from shared.validation import create_balanced_evaluation_set

# Theme labeler imports
from theme_labeler.theme_schema import ThemeFramework
from theme_labeler.frameworks.vamr import get_vamr_framework
from theme_labeler.config import ThemeClassificationConfig
from theme_labeler.zero_shot_classifier import classify_segments_zero_shot
from theme_labeler.response_parser import parse_all_results

# Codebook imports
from codebook_classifier.embedding_classifier import EmbeddingCodebookClassifier
from codebook_classifier.llm_classifier import LLMCodebookClassifier
from codebook_classifier.ensemble import CodebookEnsemble, EnsembleResult
from codebook_classifier.config import (
    EmbeddingClassifierConfig,
    LLMCodebookConfig,
    EnsembleConfig,
)
from codebook_classifier.codebooks.phenomenology import get_phenomenology_codebook

# Pipeline imports
from pipeline.config import PipelineConfig, SegmentationConfig, ValidationConfig, ConfidenceTierConfig
from pipeline.transcript_ingestion import (
    TranscriptSegmenter,
    load_diarized_session,
    discover_session_files,
)
from pipeline.dataset_assembly import (
    assemble_master_dataset,
    build_session_adjacency_index,
    export_theme_definitions,
    compute_session_stage_progression,
)
from pipeline.cross_validation import (
    compute_theme_codebook_cooccurrence,
    validate_codebook_hypothesis,
)


# =============================================================================
# Human Validation UI
# =============================================================================

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
        print(f"\n⚠️  Validation Needed: {reason}")
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
        print(f"\n⚠️  Codebook Disagreement Detected")
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
        Prompt user to review theme ↔ codebook co-occurrence anomaly.

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
        print(f"\n⚠️  Theme ↔ Codebook Anomaly Detected")
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


# =============================================================================
# Integrated Classification Pipeline
# =============================================================================

class IntegratedClassificationPipeline:
    """Orchestrates theme + codebook classification with cross-validation."""

    def __init__(
        self,
        config: PipelineConfig,
        framework: ThemeFramework,
        codebook,
        run_mode: str = 'auto',
    ):
        self.config = config
        self.framework = framework
        self.codebook = codebook
        self.run_mode = run_mode
        self.validator = HumanValidator(
            framework, codebook,
            skip_confirmation=(run_mode == 'auto'),
        )
        self.ts = datetime.datetime.now(datetime.timezone.utc).strftime('%y-%m-%dT%H-%M-%S')

    def run(self) -> pd.DataFrame:
        """Execute the complete integrated pipeline."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # ===== Stage 1: Ingestion & Segmentation =====
        print("\n" + "=" * 70)
        print("STAGE 1: Transcript Ingestion and Segmentation")
        print("=" * 70)

        all_segments = self._stage_1_ingestion()
        print(f"✓ Segmented into {len(all_segments)} segments")

        # ===== Stage 2: Operationalization =====
        print("\n" + "=" * 70)
        print("STAGE 2: Framework Operationalization")
        print("=" * 70)

        self._stage_2_operationalization()
        print("✓ Exported theme and codebook definitions")

        # ===== Stage 3: Theme Classification =====
        print("\n" + "=" * 70)
        print("STAGE 3: Theme/Stage Classification")
        print("=" * 70)

        all_segments = self._stage_3_theme_classification(all_segments)
        print("✓ Theme classification complete")

        # ===== Stage 3b: Codebook Classification =====
        print("\n" + "=" * 70)
        print("STAGE 3b: Codebook Classification")
        print("=" * 70)

        all_segments = self._stage_3b_codebook_classification(all_segments)
        print("✓ Codebook classification complete")

        # ===== Stage 4: Cross-Validation =====
        print("\n" + "=" * 70)
        print("STAGE 4: Cross-Validation (Theme ↔ Codebook)")
        print("=" * 70)

        self._stage_4_cross_validation(all_segments)
        print("✓ Cross-validation complete")

        # ===== Stage 5: Human Validation Set =====
        print("\n" + "=" * 70)
        print("STAGE 5: Human Validation Set Preparation")
        print("=" * 70)

        self._stage_5_human_validation_set(all_segments)
        print("✓ Human validation set exported")

        # ===== Stage 6: Dataset Assembly =====
        print("\n" + "=" * 70)
        print("STAGE 6: Dataset Assembly and Reporting")
        print("=" * 70)

        master_df = self._stage_6_dataset_assembly(all_segments)
        print("✓ Master dataset assembled")

        # Export validation log
        if self.validator.validation_log:
            self._export_validation_log()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"All outputs in: {self.config.output_dir}")
        return master_df

    # =========================================================================
    # Stage Implementations
    # =========================================================================

    def _stage_1_ingestion(self) -> List[Segment]:
        """Stage 1: Transcript ingestion and segmentation."""
        seg_config = {
            'embedding_model': self.config.segmentation.embedding_model,
            'min_segment_words': self.config.segmentation.min_segment_words,
            'max_segment_words': self.config.segmentation.max_segment_words,
            'silence_threshold_ms': self.config.segmentation.silence_threshold_ms,
            'semantic_shift_percentile': self.config.segmentation.semantic_shift_percentile,
        }
        segmenter = TranscriptSegmenter(seg_config)

        session_files = discover_session_files(self.config.transcript_dir)
        if not session_files:
            import glob
            session_files = sorted(glob.glob(
                os.path.join(self.config.transcript_dir, '**/*.json'), recursive=True
            ))

        all_segments: List[Segment] = []
        for session_file in session_files:
            session_data = load_diarized_session(session_file)
            metadata = session_data['metadata']
            metadata.setdefault('trial_id', self.config.trial_id)
            metadata.setdefault('participant_id', 'unknown')
            metadata.setdefault('session_id', os.path.basename(os.path.dirname(session_file)))
            metadata.setdefault('session_number', 1)

            segments = segmenter.segment_session(
                session_data['sentences'], metadata
            )
            all_segments.extend(segments)

        session_counts = Counter(s.session_id for s in all_segments)
        for seg in all_segments:
            seg.total_segments_in_session = session_counts[seg.session_id]

        return all_segments

    def _stage_2_operationalization(self):
        """Stage 2: Export framework definitions."""
        export_theme_definitions(
            self.framework,
            os.path.join(self.config.output_dir, 'theme_definitions.json'),
        )

    def _stage_3_theme_classification(self, all_segments: List[Segment]) -> List[Segment]:
        """Stage 3: Theme classification with optional human validation."""
        if not self.config.run_theme_labeler:
            print("  Skipping theme classification")
            return all_segments

        theme_config = self.config.theme_classification
        theme_config.output_dir = os.path.join(self.config.output_dir, 'llm_raw')

        # Run LLM classification
        results_all, _ = classify_segments_zero_shot(
            segments=all_segments,
            framework=self.framework,
            config=theme_config,
            resume_from=self.config.resume_from,
        )

        # Parse results
        name_to_id = self.framework.build_name_to_id_map()
        all_segments, _ = parse_all_results(
            results_all, all_segments, name_to_id
        )

        # Human validation of uncertain classifications
        if self.run_mode in ('interactive', 'review'):
            all_segments = self._validate_theme_classifications(all_segments)

        return all_segments

    def _stage_3b_codebook_classification(self, all_segments: List[Segment]) -> List[Segment]:
        """Stage 3b: Codebook classification with ensemble."""
        # Set up codebook output directory and exemplar export path
        codebook_output_dir = os.path.join(self.config.output_dir, 'codebook_raw')
        os.makedirs(codebook_output_dir, exist_ok=True)
        self.config.codebook_embedding.exemplar_export_path = os.path.join(
            codebook_output_dir, 'found_exemplar_utterances.json'
        )

        print("  Running embedding-based classification...")
        embedding_classifier = EmbeddingCodebookClassifier(
            self.config.codebook_embedding
        )
        embedding_results = embedding_classifier.classify_segments(
            all_segments, self.codebook
        )

        print("  Running LLM-based classification...")
        llm_config = LLMClientConfig(
            backend=self.config.theme_classification.backend,
            api_key=self.config.theme_classification.api_key,
            replicate_api_token=self.config.theme_classification.replicate_api_token,
            model=self.config.theme_classification.model,
        )
        llm_client = LLMClient(llm_config)
        llm_classifier = LLMCodebookClassifier(llm_client, self.config.codebook_llm)
        llm_results = llm_classifier.classify_segments(
            all_segments, self.codebook,
            output_dir=codebook_output_dir,
        )

        print("  Running ensemble reconciliation...")
        ensemble = CodebookEnsemble(EnsembleConfig(flag_disagreements=True))
        ensemble_results = ensemble.reconcile(embedding_results, llm_results)

        # Populate Segment fields and handle disagreements
        for segment in all_segments:
            if segment.segment_id in ensemble_results:
                result = ensemble_results[segment.segment_id]
                segment.codebook_labels_ensemble = result.final_codes
                segment.codebook_disagreements = [
                    d['code_id'] for d in result.disagreement_details
                ]

                # Human validation of disagreements
                if self.run_mode in ('interactive', 'review') and result.needs_human_review:
                    final_codes = self.validator.validate_codebook_disagreement(
                        segment, result
                    )
                    if final_codes is not None:
                        segment.codebook_labels_ensemble = final_codes

        return all_segments

    def _stage_4_cross_validation(self, all_segments: List[Segment]):
        """Stage 4: Cross-validate theme ↔ codebook mapping."""
        segments_df = pd.DataFrame([vars(s) for s in all_segments])

        # Compute co-occurrence
        cooccurrence = compute_theme_codebook_cooccurrence(
            segments_df, self.framework,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )  # type: Dict

        # Validate hypothesis
        validation_results = validate_codebook_hypothesis(
            cooccurrence, self.framework, min_lift=1.5
        )

        # Export results
        output_file = os.path.join(
            self.config.output_dir,
            f'cross_validation_results_{self.ts}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"  Exported cross-validation results to {os.path.basename(output_file)}")

        # Report anomalies
        self._report_cooccurrence_anomalies(segments_df, cooccurrence, validation_results)

    def _stage_5_human_validation_set(self, all_segments: List[Segment]):
        """Stage 5: Create balanced evaluation set."""
        segments_df = pd.DataFrame([vars(s) for s in all_segments])
        participant_labeled = segments_df[
            (segments_df['speaker'] == 'participant')
            & (segments_df['primary_stage'].notna())
        ]

        if len(participant_labeled) > 0:
            eval_set = create_balanced_evaluation_set(
                participant_labeled,
                n_per_class=self.config.validation.n_per_class,
            )
            eval_set.to_csv(
                os.path.join(self.config.output_dir, 'human_coding_evaluation_set.csv'),
                index=False,
            )
            print(f"  Exported {len(eval_set)} segments for human coding")

    def _stage_6_dataset_assembly(self, all_segments: List[Segment]) -> pd.DataFrame:
        """Stage 6: Assemble master dataset."""
        confidence_tier_config = asdict(self.config.confidence_tiers)
        master_df = assemble_master_dataset(
            all_segments,
            os.path.join(self.config.output_dir, f'master_segments_{self.ts}.jsonl'),
            confidence_tiers=confidence_tier_config,
        )

        build_session_adjacency_index(
            master_df,
            os.path.join(self.config.output_dir, f'session_adjacency_{self.ts}.jsonl'),
        )

        id_to_short = self.framework.build_id_to_short_map()
        progression_df = compute_session_stage_progression(master_df, id_to_short)
        if len(progression_df) > 0:
            progression_df.to_csv(
                os.path.join(self.config.output_dir, f'session_stage_progression_{self.ts}.csv'),
                index=False,
            )

        return master_df

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_theme_classifications(self, all_segments: List[Segment]) -> List[Segment]:
        """Identify and validate uncertain theme classifications."""
        uncertain = []

        for segment in all_segments:
            if segment.primary_stage is None:
                continue

            reasons = []
            if segment.llm_run_consistency and segment.llm_run_consistency < 3:
                reasons.append("inconsistency")
            if segment.llm_confidence_primary and segment.llm_confidence_primary < 0.6:
                reasons.append("low_confidence")

            if reasons:
                uncertain.append((segment, reasons))

        if not uncertain:
            print(f"  All {len(all_segments)} theme classifications are confident and consistent")
            return all_segments

        print(f"\n  Found {len(uncertain)} uncertain classifications needing validation")
        for segment, reasons in uncertain:
            result = self.validator.validate_theme_classification(
                segment,
                " + ".join(reasons),
            )
            if result:
                primary_id, secondary_id = result
                segment.primary_stage = primary_id
                segment.secondary_stage = secondary_id

        return all_segments

    def _report_cooccurrence_anomalies(
        self,
        segments_df: pd.DataFrame,
        cooccurrence: Dict,
        validation_results: Dict,
    ):
        """Identify and report theme ↔ codebook anomalies."""
        anomalies = []

        for theme_key, results in validation_results.items():
            # Unexpected strong associations
            for anomaly in results.get('unexpected', []):
                anomalies.append({
                    'theme': theme_key,
                    'code': anomaly['code'],
                    'type': 'unexpected_strong_association',
                    'lift': anomaly['lift'],
                })

            # Unconfirmed hypothesized codes
            for missing in results.get('unconfirmed', []):
                if missing['count'] > 0:  # Only if it appears at least once
                    anomalies.append({
                        'theme': theme_key,
                        'code': missing['code'],
                        'type': 'weak_association',
                        'lift': missing['lift'],
                    })

        if anomalies:
            print(f"\n  ⚠️  Found {len(anomalies)} co-occurrence anomalies")
            for anomaly in anomalies[:5]:
                print(f"    - {anomaly['theme']}: {anomaly['code']} ({anomaly['type']})")
            if len(anomalies) > 5:
                print(f"    ... and {len(anomalies) - 5} more")

            # Export anomalies
            output_file = os.path.join(
                self.config.output_dir,
                f'cooccurrence_anomalies_{self.ts}.json'
            )
            with open(output_file, 'w') as f:
                json.dump(anomalies, f, indent=2)

    def _export_validation_log(self):
        """Export human validation log."""
        output_file = os.path.join(
            self.config.output_dir,
            f'human_validation_log_{self.ts}.jsonl'
        )
        with open(output_file, 'w') as f:
            for entry in self.validator.validation_log:
                f.write(json.dumps(entry) + '\n')
        print(f"  Exported validation log: {os.path.basename(output_file)}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integrated classification and labeling pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with human validation
  python classify_and_label.py \\
      --transcript-dir ./data/diarized/ \\
      --output-dir ./data/output/ \\
      --run-mode interactive

  # Fully automated (no prompts)
  python classify_and_label.py \\
      --transcript-dir ./data/diarized/ \\
      --output-dir ./data/output/ \\
      --run-mode auto

  # With custom model and API key
  python classify_and_label.py \\
      --transcript-dir ./data/diarized/ \\
      --output-dir ./data/output/ \\
      --model openai/gpt-4o \\
      --api-key sk-or-... \\
      --run-mode interactive
        """,
    )

    # Input/output
    parser.add_argument(
        '--transcript-dir',
        default='./data/input/diarized_sessions/',
        help='Directory containing diarized session JSON files',
    )
    parser.add_argument(
        '--output-dir',
        default='./data/output/',
        help='Directory for pipeline outputs',
    )

    # Trial and model settings
    parser.add_argument(
        '--trial-id',
        default='standard',
        help='Trial identifier for this batch of sessions',
    )
    parser.add_argument(
        '--model',
        default='openai/gpt-4o',
        help='LLM model for classification',
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=3,
        help='Number of triplicate runs per segment (default: 3)',
    )

    # API settings
    parser.add_argument(
        '--backend',
        default='openrouter',
        choices=['openrouter', 'replicate'],
        help='API backend for LLM (default: openrouter)',
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)',
    )
    parser.add_argument(
        '--replicate-api-token',
        default=None,
        help='Replicate API token (or set REPLICATE_API_TOKEN env var)',
    )

    # Confidence settings
    parser.add_argument(
        '--high-confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for high-tier label (default: 0.8)',
    )
    parser.add_argument(
        '--medium-confidence-threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for medium-tier label (default: 0.6)',
    )

    # Checkpoint/resumption
    parser.add_argument(
        '--resume-from',
        default=None,
        help='Path to checkpoint JSON file to resume from',
    )

    # Validation mode
    parser.add_argument(
        '--run-mode',
        default='auto',
        choices=['auto', 'interactive', 'review'],
        help=(
            'auto: fully automated; '
            'interactive: prompt for validation as you go; '
            'review: batch validation at end'
        ),
    )

    # Feature flags
    parser.add_argument(
        '--no-theme-labeler',
        action='store_true',
        help='Skip theme classification',
    )
    parser.add_argument(
        '--no-codebook-classifier',
        action='store_true',
        help='Skip codebook classification',
    )

    # Embedding classifier / two-pass settings
    parser.add_argument(
        '--no-two-pass',
        action='store_true',
        help='Disable two-pass embedding classification (single pass only)',
    )
    parser.add_argument(
        '--exemplar-import-path',
        default=None,
        help='Path to JSON file with pre-populated exemplar utterances per code',
    )
    parser.add_argument(
        '--criteria-weight',
        type=float,
        default=0.5,
        help='Weight for criteria similarity in embedding scoring (default: 0.5)',
    )
    parser.add_argument(
        '--exemplar-weight',
        type=float,
        default=0.5,
        help='Weight for exemplar similarity in embedding scoring (default: 0.5)',
    )
    parser.add_argument(
        '--exemplar-confidence-threshold',
        type=float,
        default=0.8,
        help='Minimum confidence for a segment to become an exemplar in pass 2 (default: 0.8)',
    )
    parser.add_argument(
        '--max-exemplar-tokens',
        type=int,
        default=512,
        help='Maximum word count for combined exemplar text per code (default: 512)',
    )

    args = parser.parse_args()

    # Resolve API credentials
    if args.backend == 'openrouter':
        api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY', '')
        if not api_key:
            print("Error: No API key provided. Set OPENROUTER_API_KEY or use --api-key")
            sys.exit(1)
        replicate_token = ''
    else:
        api_key = ''
        replicate_token = args.replicate_api_token or os.environ.get('REPLICATE_API_TOKEN', '')
        if not replicate_token:
            print("Error: No Replicate token provided. Set REPLICATE_API_TOKEN or use --replicate-api-token")
            sys.exit(1)

    # Build configuration
    config = PipelineConfig(
        transcript_dir=args.transcript_dir,
        trial_id=args.trial_id,
        output_dir=args.output_dir,
        run_theme_labeler=not args.no_theme_labeler,
        run_codebook_classifier=not args.no_codebook_classifier,
        segmentation=SegmentationConfig(),
        theme_classification=ThemeClassificationConfig(
            model=args.model,
            n_runs=args.n_runs,
            api_key=api_key,
            backend=args.backend,
            replicate_api_token=replicate_token,
        ),
        codebook_embedding=EmbeddingClassifierConfig(
            two_pass=not args.no_two_pass,
            exemplar_import_path=args.exemplar_import_path,
            criteria_weight=args.criteria_weight,
            exemplar_weight=args.exemplar_weight,
            exemplar_confidence_threshold=args.exemplar_confidence_threshold,
            max_exemplar_tokens=args.max_exemplar_tokens,
        ),
        validation=ValidationConfig(),
        confidence_tiers=ConfidenceTierConfig(
            high_confidence=args.high_confidence_threshold,
            medium_min_confidence=args.medium_confidence_threshold,
        ),
        resume_from=args.resume_from,
    )

    # Get frameworks and codebooks
    framework = get_vamr_framework()
    codebook = get_phenomenology_codebook()

    print("\n" + "=" * 70)
    print("INTEGRATED CLASSIFICATION AND LABELING PIPELINE")
    print("=" * 70)
    print(f"Mode: {args.run_mode.upper()}")
    print(f"Framework: {framework.name}")
    print(f"Codebook: {codebook.name}")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")

    # Run pipeline
    pipeline = IntegratedClassificationPipeline(
        config, framework, codebook, run_mode=args.run_mode
    )
    master_df = pipeline.run()

    print(f"\nResults shape: {master_df.shape}")
    print(f"Theme labels: {master_df['primary_stage'].notna().sum()} segments")
    print(f"Codebook labels: {master_df['codebook_labels_ensemble'].notna().sum()} segments")


if __name__ == '__main__':
    main()
