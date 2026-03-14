"""
vamr_labeling: Zero-Shot Data Labeling Pipeline for VA-MR Classification

This package implements the complete pipeline from diarized transcripts
to validated labeled datasets for downstream consumption by AutoResearch
(Mindful-BERT fine-tuning) and CFiCS (graph-based analysis).

Pipeline stages:
    1. Transcript ingestion and segment boundary detection
    2. Construct operationalization (VA-MR + phenomenology codebook)
    3. Zero-shot LLM classification with triplicate-and-flag
    4. Response parsing and error handling
    5. Human validation and adjudication
    6. Dataset assembly (5 output files)

Adapted from the Text Psychometrics codebase (Low et al.) for the
Vigilance-Avoidance Metacognition-Reappraisal (VA-MR) framework.
"""

__version__ = "0.1.0"
