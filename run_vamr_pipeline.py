#!/usr/bin/env python3
"""
run_vamr_pipeline.py
--------------------
CLI entry point for the VA-MR zero-shot labeling pipeline.

Usage:
    # Run the full pipeline
    python run_vamr_pipeline.py --transcript-dir ./data/diarized/ --output-dir ./data/output/

    # Specify trial and model
    python run_vamr_pipeline.py \
        --transcript-dir ./data/diarized/ \
        --trial-id standard_MORE \
        --model openai/gpt-4o \
        --n-runs 3

    # Generate validation report after human coding
    python run_vamr_pipeline.py report \
        --master-dataset ./data/output/master_segments.jsonl \
        --human-codes ./data/output/human_coded.csv
"""

import argparse
import os
import sys

from vamr_labeling.config import PipelineConfig, SegmentationConfig, ClassificationConfig, ValidationConfig
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
        help='LLM model for zero-shot classification (OpenRouter model ID)',
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

    args = parser.parse_args()

    # Build config
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        print("Error: No API key provided. Set OPENROUTER_API_KEY or use --api-key")
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
        ),
        validation=ValidationConfig(),
    )

    run_full_pipeline(config)


if __name__ == '__main__':
    main()
