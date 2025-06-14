import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import srt
from collections import Counter
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
import json
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MindfulBERTCoder:
    """
    Applies qualitative codes using a fine-tuned MindfulBERT classifier
    """
    
    def __init__(self, 
                 model_path: str = "models/mindful-bert",
                 codebook_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the MindfulBERT coder
        
        Args:
            model_path: Path to fine-tuned MindfulBERT model
            codebook_path: Path to codebook JSON file
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the fine-tuned model and tokenizer
        logger.info(f"Loading MindfulBERT from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load codebook to get label mappings
        if codebook_path:
            self.load_codebook(codebook_path)
        else:
            # Will be set when apply_research_codes_to_sentences is called
            self.codebook = None
            self.label2id = None
            self.id2label = None
    
    def load_codebook(self, codebook_path: str):
        """Load codebook and create label mappings"""
        with open(codebook_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            self.codebook = data
        else:
            self.codebook = data.get('codes', data)
            
        # Create label mappings
        self.label2id = {code['category']: i for i, code in enumerate(self.codebook)}
        self.id2label = {i: code['category'] for i, code in enumerate(self.codebook)}
        
        # Verify model config matches codebook
        if hasattr(self.model.config, 'num_labels'):
            expected_labels = len(self.codebook)
            if self.model.config.num_labels != expected_labels:
                logger.warning(f"Model expects {self.model.config.num_labels} labels but codebook has {expected_labels}")
    
    def predict_codes(self, 
                     text: str, 
                     threshold: float = 0.5,
                     max_codes: int = 5) -> List[Tuple[str, float]]:
        """
        Predict codes for a given text segment
        
        Args:
            text: Text to classify
            threshold: Probability threshold for applying codes
            max_codes: Maximum number of codes to apply
            
        Returns:
            List of (code_name, probability) tuples
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Get codes above threshold
        predictions = []
        for idx, prob in enumerate(probabilities):
            if prob >= threshold and idx in self.id2label:
                predictions.append((self.id2label[idx], float(prob)))
        
        # Sort by probability and limit to max_codes
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:max_codes]
    
    def apply_codes_to_srt(self,
                          srt_file: str,
                          output_path: Optional[str] = None,
                          threshold: float = 0.5,
                          max_codes_per_segment: int = 3,
                          remove_instructor: bool = True,
                          code_instructor: bool = False,
                          coded_output_only: bool = False) -> Tuple[str, Dict, List]:
        """
        Apply codes to an SRT file
        
        Args:
            srt_file: Path to SRT file
            output_path: Optional path to save coded SRT
            threshold: Probability threshold for codes
            max_codes_per_segment: Maximum codes per segment
            remove_instructor: Remove instructor segments from output
            code_instructor: Whether to code instructor segments
            coded_output_only: Only output coded segments
            
        Returns:
            Tuple of (coded_srt_string, statistics, codes_applied_list)
        """
        # Load SRT file
        with open(srt_file, 'r', encoding='utf-8') as file:
            subtitle_generator = srt.parse(file.read())
            subtitle_list = list(subtitle_generator)
        
        # Statistics tracking
        codes_applied_list = []
        total_segments = 0
        coded_segments = 0
        
        # Process each subtitle
        for i, subtitle in enumerate(tqdm(subtitle_list, desc="Coding segments")):
            # Check if instructor segment
            is_instructor = "instructor:" in subtitle.content.lower() or \
                          "main_speaker:" in subtitle.content.lower()
            
            # Skip instructor segments if requested
            if is_instructor and not code_instructor:
                if remove_instructor:
                    subtitle.content = "..."
                continue
            
            # Skip very short segments
            if len(subtitle.content.split()) < 5:
                continue
                
            total_segments += 1
            
            # Get predictions
            predictions = self.predict_codes(
                subtitle.content,
                threshold=threshold,
                max_codes=max_codes_per_segment
            )
            
            if predictions:
                coded_segments += 1
                # Format codes for subtitle
                code_names = [code for code, prob in predictions]
                codes_applied_list.extend(code_names)
                
                # Add codes to subtitle content
                if not is_instructor or code_instructor:
                    codes_str = " ~ == ".join([""] + code_names)
                    subtitle.content = subtitle.content + "\n" + codes_str
        
        # Filter for coded output only if requested
        if coded_output_only:
            subtitle_list = [sub for sub in subtitle_list if "~ == " in sub.content]
        
        # Compose SRT
        coded_srt = srt.compose(subtitle_list)
        
        # Calculate statistics
        code_counts = Counter(codes_applied_list)
        stats = {
            'total_segments': total_segments,
            'coded_segments': coded_segments,
            'total_codes_applied': len(codes_applied_list),
            'unique_codes_applied': len(set(codes_applied_list)),
            'coding_rate': coded_segments / total_segments if total_segments > 0 else 0,
            'avg_codes_per_segment': len(codes_applied_list) / coded_segments if coded_segments > 0 else 0,
            'code_frequencies': dict(code_counts.most_common())
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(coded_srt)
            logger.info(f"Saved coded SRT to {output_path}")
        
        return coded_srt, stats, codes_applied_list


def apply_research_codes_to_sentences(srt_file: str,
                                     codes: List[Dict],
                                     model_path: str = "models/mindful-bert",
                                     threshold: float = 0.5,
                                     max_codes_per_sentence: Optional[int] = None,
                                     remove_instructor: bool = True,
                                     code_instructor: bool = False,
                                     coded_output_only: bool = False,
                                     **kwargs) -> Tuple[str, Dict, List]:
    """
    Apply research codes to sentences using fine-tuned MindfulBERT
    
    This function maintains compatibility with the original interface
    while using the new classification approach
    
    Args:
        srt_file: Path to SRT file
        codes: List of code dictionaries (used for compatibility)
        model_path: Path to fine-tuned model
        threshold: Probability threshold for applying codes
        max_codes_per_sentence: Maximum codes per sentence
        remove_instructor: Remove instructor speech from output
        code_instructor: Whether to code instructor speech
        coded_output_only: Only output coded segments
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Tuple of (coded_srt, stats, codes_applied_list)
    """
    # Initialize coder
    coder = MindfulBERTCoder(model_path=model_path)
    
    # Set codebook from provided codes
    coder.codebook = codes
    coder.label2id = {code['category']: i for i, code in enumerate(codes)}
    coder.id2label = {i: code['category'] for i, code in enumerate(codes)}
    
    # Set max codes
    if max_codes_per_sentence is None:
        max_codes_per_sentence = min(6, int(len(codes) * 0.33))
    
    # Apply codes
    return coder.apply_codes_to_srt(
        srt_file=srt_file,
        threshold=threshold,
        max_codes_per_segment=max_codes_per_sentence,
        remove_instructor=remove_instructor,
        code_instructor=code_instructor,
        coded_output_only=coded_output_only
    )


def batch_apply_codes(srt_files: List[str],
                     codes: List[Dict],
                     model_path: str = "models/mindful-bert",
                     output_dir: str = "coded_output",
                     threshold: float = 0.5,
                     **kwargs) -> Dict:
    """
    Apply codes to multiple SRT files in batch
    
    Args:
        srt_files: List of SRT file paths
        codes: Codebook
        model_path: Path to fine-tuned model
        output_dir: Directory for output files
        threshold: Classification threshold
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with aggregated statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize coder once for efficiency
    coder = MindfulBERTCoder(model_path=model_path)
    coder.codebook = codes
    coder.label2id = {code['category']: i for i, code in enumerate(codes)}
    coder.id2label = {i: code['category'] for i, code in enumerate(codes)}
    
    # Process files
    all_results = []
    all_codes_applied = []
    
    for srt_file in tqdm(srt_files, desc="Processing files"):
        output_path = os.path.join(output_dir, 
                                  f"coded_{os.path.basename(srt_file)}")
        
        coded_srt, stats, codes_applied = coder.apply_codes_to_srt(
            srt_file=srt_file,
            output_path=output_path,
            threshold=threshold,
            **kwargs
        )
        
        all_results.append({
            'file': srt_file,
            'stats': stats,
            'output_path': output_path
        })
        all_codes_applied.extend(codes_applied)
    
    # Aggregate statistics
    total_stats = {
        'files_processed': len(srt_files),
        'total_segments': sum(r['stats']['total_segments'] for r in all_results),
        'total_coded_segments': sum(r['stats']['coded_segments'] for r in all_results),
        'total_codes_applied': len(all_codes_applied),
        'overall_code_frequencies': dict(Counter(all_codes_applied).most_common()),
        'file_results': all_results
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "coding_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info(f"Batch processing complete. Summary saved to {summary_path}")
    return total_stats


# Compatibility functions for specific codebooks
def get_codebook(which=None):
    """
    Load predefined codebooks (for compatibility)
    """
    # This function can remain the same or be simplified
    # since the fine-tuned model already knows the codes
    logger.warning("get_codebook is deprecated when using fine-tuned models")
    return []


def main():
    """
    Example usage and testing
    """
    # Example of using the new approach
    print("=== MindfulBERT Coding Example ===\n")
    
    # Initialize coder
    coder = MindfulBERTCoder(
        model_path="models/mindful-bert",
        codebook_path="codebooks/more_codebook.json"
    )
    
    # Test on sample text
    sample_text = "I noticed my mind wandering to my back pain during the meditation."
    predictions = coder.predict_codes(sample_text)
    
    print(f"Text: {sample_text}")
    print("Predicted codes:")
    for code, prob in predictions:
        print(f"  - {code}: {prob:.3f}")
    
    # Process SRT files if available
    if os.path.exists("more_transcripts"):
        srt_files = list(Path("more_transcripts").glob("*.srt"))
        if srt_files:
            print(f"\nProcessing {len(srt_files)} transcript files...")
            
            # Load codebook
            with open("codebooks/more_codebook.json", 'r') as f:
                codebook = json.load(f).get('codes', [])
            
            # Batch process
            results = batch_apply_codes(
                srt_files=srt_files,
                codes=codebook,
                threshold=0.5,
                max_codes_per_segment=3
            )
            
            print("\nProcessing complete!")
            print(f"Total segments coded: {results['total_coded_segments']}")
            print(f"Total codes applied: {results['total_codes_applied']}")
            print("\nTop 5 most frequent codes:")
            for code, count in list(results['overall_code_frequencies'].items())[:5]:
                print(f"  - {code}: {count}")


if __name__ == "__main__":
    main()
