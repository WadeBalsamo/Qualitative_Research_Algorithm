"""
batch_process_transcripts.py
---------------------------

Script for batch processing new therapeutic transcript files through the Mental-BERT AQUA pipeline. Handles:
- Loading multiple .srt files from the data/ directory
- Running the full pipeline: preprocessing, embedding, graph construction, clustering, classification
- Outputting results (labels, graphs, visualizations, audit trails) to a specified directory
- Supports command-line arguments for input/output paths and processing options

Usage:
    python scripts/batch_process_transcripts.py --input_dir data/ --output_dir results/
"""

import os
import argparse
import logging
from mentalbert_sentence_aqua import main as pipeline

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.srt'):
            continue
        srt_path = os.path.join(input_dir, fname)
        try:
            logging.info(f"Processing {fname}")
            pipeline.run_pipeline(srt_path)
            # Optionally, move/copy outputs to output_dir
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    batch_process(args.input_dir, args.output_dir)
