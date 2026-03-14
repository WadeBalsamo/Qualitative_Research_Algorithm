#!/usr/bin/env python3
"""
run_vamr_pipeline.py
--------------------
CLI entry point for the VA-MR zero-shot labeling pipeline.

Usage:
    # Run the full pipeline with OpenRouter
    python run_vamr_pipeline.py --transcript-dir ./data/diarized/ --output-dir ./data/output/

    # Specify trial and model
    python run_vamr_pipeline.py \
        --transcript-dir ./data/diarized/ \
        --trial-id standard_MORE \
        --model openai/gpt-4o \
        --n-runs 3

    # Use Replicate backend for open-source models
    python run_vamr_pipeline.py \
        --backend replicate \
        --model google-deepmind/gemma-2b \
        --replicate-api-token $REPLICATE_API_TOKEN

    # Resume from a checkpoint
    python run_vamr_pipeline.py \
        --resume-from ./data/output/llm_raw/llm_results_openai_gpt-4o_*.json

    # Custom confidence thresholds
    python run_vamr_pipeline.py \
        --high-confidence-threshold 0.85 \
        --medium-confidence-threshold 0.65
"""

import argparse
import os
import sys

from vamr_labeling.config import (
    PipelineConfig, SegmentationConfig, ClassificationConfig,
    ValidationConfig, ConfidenceTierConfig,
)
from vamr_labeling.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="VA-MR Zero-Shot Data Labeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
    parser.add_argument(
        '--trial-id',
        default='standard_MORE',
        choices=['standard_MORE', 'abbreviated_MORE', 'move_MORE', 'STAMP'],
        help='Trial identifier for this batch of sessions',
    )
    parser.add_argument(
        '--model',
        default='openai/gpt-4o',
        help='LLM model for classification (OpenRouter or Replicate model ID)',
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=3,
        help='Number of triplicate runs per segment (default: 3)',
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)',
    )
    parser.add_argument(
        '--backend',
        default='openrouter',
        choices=['openrouter', 'replicate'],
        help='API backend for LLM classification (default: openrouter)',
    )
    parser.add_argument(
        '--replicate-api-token',
        default=None,
        help='Replicate API token (or set REPLICATE_API_TOKEN env var)',
    )
    parser.add_argument(
        '--resume-from',
        default=None,
        help='Path to checkpoint JSON file to resume classification from',
    )
    parser.add_argument(
        '--high-confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for high-tier label assignment (default: 0.8)',
    )
    parser.add_argument(
        '--medium-confidence-threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for medium-tier label assignment (default: 0.6)',
    )

    args = parser.parse_args()

    # Build config
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

    config = PipelineConfig(
        transcript_dir=args.transcript_dir,
        trial_id=args.trial_id,
        output_dir=args.output_dir,
        segmentation=SegmentationConfig(),
        classification=ClassificationConfig(
            model=args.model,
            n_runs=args.n_runs,
            api_key=api_key,
            backend=args.backend,
            replicate_api_token=replicate_token,
        ),
        validation=ValidationConfig(),
        confidence_tiers=ConfidenceTierConfig(
            high_confidence=args.high_confidence_threshold,
            medium_min_confidence=args.medium_confidence_threshold,
        ),
        resume_from=args.resume_from,
    )

    run_full_pipeline(config)


if __name__ == '__main__':
    main()
