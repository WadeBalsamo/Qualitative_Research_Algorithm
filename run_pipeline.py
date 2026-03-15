#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Main entry point for the QRA classification pipeline using three locally-hosted
Hugging Face models with cross-referencing:
  - LLAMA 4 Maverick 17B (meta-llama/Llama-4-Maverick-17B-128E-Instruct)
  - Mixtral 8x7B Instruct (mistralai/Mixtral-8x7B-Instruct-v0.1)
  - Qwen 3 Next 80B (Qwen/Qwen3-Next-80B-A3B-Instruct)

This script:
1. Downloads models from Hugging Face if not found locally
2. Loads models into memory
3. Runs the full classification pipeline with multi-model cross-referencing
4. Outputs results to the configured output directory

Configuration is via CLI arguments (see --help) with sensible defaults.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.model_loader import (
    ensure_models_ready,
    load_model,
    LLAMA_MAVERICK_MODEL,
    MIXTRAL_MODEL,
    QWEN_MODEL,
)
from pipeline.config import (
    PipelineConfig,
    SegmentationConfig,
    ValidationConfig,
    ConfidenceTierConfig,
)
from pipeline.orchestrator import run_full_pipeline
from theme_labeler.config import ThemeClassificationConfig
from theme_labeler.frameworks.vamr import get_vamr_framework
from codebook_classifier.config import EmbeddingClassifierConfig


# Three Hugging Face models for cross-referencing
MODELS = [
    LLAMA_MAVERICK_MODEL,  # "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    MIXTRAL_MODEL,         # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    QWEN_MODEL,            # "Qwen/Qwen3-Next-80B-A3B-Instruct"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="QRA Multi-Model Classification Pipeline (Hugging Face)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run theme-only classification (default)
  python run_pipeline.py

  # Run with codebook classification enabled
  python run_pipeline.py --run-codebook-classifier

  # Run with two-pass embedding disabled
  python run_pipeline.py --run-codebook-classifier --no-two-pass

  # Supply pre-populated exemplars
  python run_pipeline.py --run-codebook-classifier \\
      --exemplar-import-path ./exemplars.json

  # Custom directories and trial ID
  python run_pipeline.py \\
      --transcript-dir ./my_data/ \\
      --output-dir ./my_output/ \\
      --trial-id my_experiment
        """,
    )

    # Input/output
    parser.add_argument(
        '--transcript-dir',
        default='./data/input/diarized_sessions/',
        help='Directory containing diarized session JSON files (default: ./data/input/diarized_sessions/)',
    )
    parser.add_argument(
        '--output-dir',
        default='./data/output/',
        help='Directory for pipeline outputs (default: ./data/output/)',
    )
    parser.add_argument(
        '--trial-id',
        default='multi_model_trial',
        help='Trial identifier for this batch (default: multi_model_trial)',
    )

    # Classification settings
    parser.add_argument(
        '--n-runs',
        type=int,
        default=3,
        help='Number of classification runs per model per segment (default: 3)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature for classification (default: 0.0)',
    )

    # Confidence thresholds
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

    # Feature flags
    parser.add_argument(
        '--no-theme-labeler',
        action='store_true',
        help='Skip theme classification',
    )
    parser.add_argument(
        '--run-codebook-classifier',
        action='store_true',
        help='Enable codebook classification (embedding + LLM ensemble)',
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

    # Resume
    parser.add_argument(
        '--resume-from',
        default=None,
        help='Path to checkpoint JSON file to resume from',
    )

    return parser.parse_args()


def main():
    """
    Main pipeline execution.

    Steps:
    1. Download and verify Hugging Face models
    2. Load models into memory
    3. Configure the pipeline
    4. Run the full classification pipeline
    """
    args = parse_args()

    print("\n" + "=" * 70)
    print("QRA MULTI-MODEL CLASSIFICATION PIPELINE (Hugging Face)")
    print("=" * 70)

    # Step 1: Ensure models are downloaded
    print("\nStep 1: Downloading and verifying models...")
    ensure_models_ready(download_if_missing=True)

    # Step 2: Pre-load models to verify they work
    print("\nStep 2: Pre-loading models into memory...")
    for model_id in MODELS:
        try:
            print(f"  Loading {model_id}...")
            load_model(model_id)
            print(f"  Loaded successfully")
        except Exception as e:
            print(f"  Failed to load {model_id}: {e}")
            print("  Please check the model name and try again.")
            sys.exit(1)

    # Step 3: Configure the pipeline
    print("\nStep 3: Configuring pipeline...")
    print(f"  Transcript directory: {args.transcript_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Trial ID: {args.trial_id}")
    print(f"  Backend: Hugging Face (local)")
    print(f"  Models: {len(MODELS)}")
    for model in MODELS:
        print(f"    - {model}")
    print(f"  Runs per model: {args.n_runs}")
    if args.run_codebook_classifier:
        print(f"  Codebook classifier: ENABLED")
        print(f"    Two-pass: {'disabled' if args.no_two_pass else 'enabled'}")
        print(f"    Criteria weight: {args.criteria_weight}")
        print(f"    Exemplar weight: {args.exemplar_weight}")
        if args.exemplar_import_path:
            print(f"    Exemplar import: {args.exemplar_import_path}")

    config = PipelineConfig(
        transcript_dir=args.transcript_dir,
        trial_id=args.trial_id,
        output_dir=args.output_dir,
        run_theme_labeler=not args.no_theme_labeler,
        run_codebook_classifier=args.run_codebook_classifier,
        segmentation=SegmentationConfig(),
        theme_classification=ThemeClassificationConfig(
            backend='huggingface',
            model=MODELS[0],  # Primary model (fallback)
            models=MODELS,    # All models for cross-referencing
            n_runs=args.n_runs,
            temperature=args.temperature,
            randomize_codebook=True,
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

    # Step 4: Load framework (VA-MR is default)
    framework = get_vamr_framework()
    print(f"\nStep 4: Using framework: {framework.name}")
    print(f"  Themes: {framework.num_themes}")

    # Step 5: Run the pipeline
    print("\n" + "=" * 70)
    print("STARTING CLASSIFICATION PIPELINE")
    print("=" * 70 + "\n")

    master_df = run_full_pipeline(config, framework)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults written to: {args.output_dir}")
    print(f"Total segments processed: {len(master_df)}")

    # Summary statistics
    if 'primary_stage' in master_df.columns:
        labeled = master_df[master_df['primary_stage'].notna()]
        print(f"Successfully labeled: {len(labeled)}")

        if 'model_agreement' in master_df.columns:
            agreement_counts = master_df['model_agreement'].value_counts()
            print("\nModel agreement breakdown:")
            for agreement_type, count in agreement_counts.items():
                print(f"  {agreement_type}: {count}")

    print("\nNext steps:")
    print("  1. Review multi-model agreement patterns in master_segments_*.jsonl")
    print("  2. Examine disagreement cases for quality assessment")
    print("  3. Complete human validation of flagged segments")

    return master_df


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
