#!/usr/bin/env python3
"""
main.py - Main execution script for MORE Qualitative Analysis Pipeline
Provides command-line interface for clinical researchers
"""

import argparse
import yaml
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import traceback

# Import our modules
from generate_codebook import LlamaCodebookGenerator
from clinical_analysis_tools import ClinicalAnalysisTools
from apply_codebook import apply_research_codes_to_sentences

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'more_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MOREAnalysisPipeline:
    """
    Main pipeline for MORE qualitative analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self._setup_directories()
        
        # Initialize components
        logger.info("Initializing analysis pipeline...")
        self.generator = LlamaCodebookGenerator(
            model_path=self.config['models']['llama']['model_path'],
            clinical_bert=self.config['models']['bert']['use_clinical'],
            quantization=self.config['models']['llama']['quantization']
        )
        self.clinical_tools = ClinicalAnalysisTools(self.generator)
        
        # Load theoretical domains into generator
        self.generator.theoretical_domains = self.config['theoretical_domains']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _setup_directories(self):
        """Create necessary directories"""
        for key, path in self.config['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def run_inductive_generation(self, args):
        """Run inductive codebook generation"""
        logger.info("Starting inductive codebook generation...")
        
        # Load transcripts
        transcripts = self.generator.load_transcripts(self.config['paths']['transcript_dir'])
        logger.info(f"Loaded {len(transcripts)} transcript files")
        
        # Generate codebook
        codebook = self.generator.generate_inductive_codebook(
            transcripts,
            n_codes=self.config['analysis']['n_codes'],
            interactive_refinement=self.config['analysis']['interactive_refinement']
        )
        
        # Save codebook
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        codebook_path = os.path.join(
            self.config['paths']['codebook_dir'],
            f'inductive_codebook_{timestamp}.json'
        )
        self.generator.save_codebook(codebook_path)
        
        # Apply codebook if requested
        if args.apply:
            logger.info("Applying generated codebook to transcripts...")
            coded_results = self.generator.apply_codebook_to_transcripts(
                transcripts,
                output_dir=os.path.join(self.config['paths']['output_dir'], f'coded_{timestamp}')
            )
            
            # Generate report if requested
            if args.report:
                self._generate_reports(codebook, coded_results, timestamp)
                
        logger.info("Inductive generation complete!")
        return codebook
    
    def run_deductive_coding(self, args):
        """Run deductive coding with existing codebook"""
        logger.info("Starting deductive coding...")
        
        # Load codebook
        if not os.path.exists(args.codebook):
            logger.error(f"Codebook file not found: {args.codebook}")
            sys.exit(1)
            
        self.generator.load_codebook(args.codebook)
        logger.info(f"Loaded codebook with {len(self.generator.codebook)} codes")
        
        # Load transcripts
        transcripts = self.generator.load_transcripts(self.config['paths']['transcript_dir'])
        
        # Apply codebook
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        coded_results = self.generator.apply_codebook_to_transcripts(
            transcripts,
            output_dir=os.path.join(self.config['paths']['output_dir'], f'deductive_{timestamp}')
        )
        
        # Generate report if requested
        if args.report:
            self._generate_reports(self.generator.codebook, coded_results, timestamp)
            
        logger.info("Deductive coding complete!")
        return coded_results
    
    def run_iterative_refinement(self, args):
        """Run iterative inductive-deductive-inductive refinement"""
        logger.info("Starting iterative refinement process...")
        
        # Load transcripts
        transcripts = self.generator.load_transcripts(self.config['paths']['transcript_dir'])
        
        # Run iterative refinement
        refined_codebook = self.generator.iterative_refinement(
            transcripts,
            n_iterations=self.config['analysis']['n_iterations']
        )
        
        # Save final codebook
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        codebook_path = os.path.join(
            self.config['paths']['codebook_dir'],
            f'refined_codebook_{timestamp}.json'
        )
        self.generator.save_codebook(codebook_path)
        
        # Final coding pass
        coded_results = self.generator.apply_codebook_to_transcripts(
            transcripts,
            output_dir=os.path.join(self.config['paths']['output_dir'], f'refined_{timestamp}')
        )
        
        # Generate comprehensive report
        self._generate_reports(refined_codebook, coded_results, timestamp)
        
        logger.info("Iterative refinement complete!")
        return refined_codebook
    
    def run_validation(self, args):
        """Run validation against human coding"""
        logger.info("Starting validation process...")
        
        if not os.path.exists(args.human_codes):
            logger.error(f"Human coded file not found: {args.human_codes}")
            sys.exit(1)
            
        if not os.path.exists(args.ai_codes):
            logger.error(f"AI coded file not found: {args.ai_codes}")
            sys.exit(1)
            
        # Run validation
        reliability = self.clinical_tools.validate_inter_rater_reliability(
            args.human_codes,
            args.ai_codes
        )
        
        # Check against thresholds
        passed_kappa = reliability['cohen_kappa'] >= self.config['validation']['min_kappa']
        passed_agreement = reliability['percent_agreement'] >= self.config['validation']['min_agreement']
        
        # Print results
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(f"Cohen's Kappa: {reliability['cohen_kappa']:.3f} " + 
              ("✓" if passed_kappa else "✗"))
        print(f"Percent Agreement: {reliability['percent_agreement']:.1%} " + 
              ("✓" if passed_agreement else "✗"))
        print(f"N Segments: {reliability['n_segments']}")
        print("\nCategory-specific agreement:")
        
        for category, agreement in sorted(reliability['category_agreement'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {category}: {agreement:.1%}")
            
        # Save validation report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            self.config['paths']['reports_dir'],
            f'validation_report_{timestamp}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(reliability, f, indent=2)
            
        logger.info(f"Validation report saved to {report_path}")
        return reliability
    
    def _generate_reports(self, codebook, coded_results, timestamp):
        """Generate clinical analysis reports"""
        logger.info("Generating clinical analysis reports...")
        
        # PDF report
        pdf_path = os.path.join(
            self.config['paths']['reports_dir'],
            f'clinical_report_{timestamp}.pdf'
        )
        self.clinical_tools.generate_clinical_report(
            codebook,
            coded_results,
            pdf_path
        )
        
        # Export codebook in multiple formats
        for format in self.config['output']['formats']:
            try:
                export_path = self.clinical_tools.export_for_clinical_use(
                    codebook,
                    output_format=format
                )
                logger.info(f"Exported {format} to {export_path}")
            except Exception as e:
                logger.warning(f"Failed to export {format}: {e}")
    
    def run_cue_response_analysis(self, args):
        """Run CueResponse analysis for specific sessions"""
        logger.info("Starting CueResponse analysis...")
        
        # Load transcripts
        transcripts = self.generator.load_transcripts(self.config['paths']['transcript_dir'])
        
        # Initialize storage for cue-response patterns
        cue_response_patterns = []
        
        for session_id, subtitles in transcripts.items():
            logger.info(f"Analyzing session {session_id}...")
            
            # Separate speakers
            instructor_texts, participant_texts = self.generator.separate_speakers(subtitles)
            
            # Generate summaries
            if instructor_texts and participant_texts:
                instructor_summary = self.generator.generate_session_summary(
                    instructor_texts, "instructor"
                )
                participant_summary = self.generator.generate_session_summary(
                    participant_texts, "participant"
                )
                
                # Extract key cues and responses
                instructor_cues = self.generator.extract_key_concepts(instructor_summary, 5)
                participant_responses = self.generator.extract_key_concepts(participant_summary, 5)
                
                cue_response_patterns.append({
                    'session_id': session_id,
                    'instructor_cues': instructor_cues,
                    'participant_responses': participant_responses,
                    'instructor_summary': instructor_summary,
                    'participant_summary': participant_summary
                })
        
        # Save analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = os.path.join(
            self.config['paths']['output_dir'],
            f'cue_response_analysis_{timestamp}.json'
        )
        
        with open(analysis_path, 'w') as f:
            json.dump(cue_response_patterns, f, indent=2)
            
        logger.info(f"CueResponse analysis saved to {analysis_path}")
        
        # Generate summary report
        self._generate_cue_response_report(cue_response_patterns, timestamp)
        
        return cue_response_patterns
    
    def _generate_cue_response_report(self, patterns, timestamp):
        """Generate report for CueResponse analysis"""
        report_lines = [
            "CUE-RESPONSE ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sessions Analyzed: {len(patterns)}",
            "",
            "TOP INSTRUCTOR CUES:",
            "-" * 30
        ]
        
        # Aggregate all cues
        all_cues = []
        for pattern in patterns:
            all_cues.extend(pattern['instructor_cues'])
        
        # Count frequencies
        from collections import Counter
        cue_counts = Counter(all_cues)
        
        for cue, count in cue_counts.most_common(10):
            report_lines.append(f"• {cue} (n={count})")
            
        report_lines.extend([
            "",
            "TOP PARTICIPANT RESPONSES:",
            "-" * 30
        ])
        
        # Aggregate all responses
        all_responses = []
        for pattern in patterns:
            all_responses.extend(pattern['participant_responses'])
            
        response_counts = Counter(all_responses)
        
        for response, count in response_counts.most_common(10):
            report_lines.append(f"• {response} (n={count})")
            
        # Save report
        report_path = os.path.join(
            self.config['paths']['reports_dir'],
            f'cue_response_report_{timestamp}.txt'
        )
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logger.info(f"CueResponse report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MORE Qualitative Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate inductive codebook
  python main.py inductive --apply --report
  
  # Apply existing codebook
  python main.py deductive --codebook codebooks/my_codebook.json --report
  
  # Run iterative refinement
  python main.py iterative
  
  # Validate against human coding
  python main.py validate --human-codes human_coded.json --ai-codes ai_coded.json
  
  # Run CueResponse analysis
  python main.py cue-response
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['inductive', 'deductive', 'iterative', 'validate', 'cue-response'],
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--codebook',
        help='Path to existing codebook (for deductive mode)'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply generated codebook to transcripts'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate clinical analysis report'
    )
    
    parser.add_argument(
        '--human-codes',
        help='Path to human coded file (for validation)'
    )
    
    parser.add_argument(
        '--ai-codes',
        help='Path to AI coded file (for validation)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    try:
        pipeline = MOREAnalysisPipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == 'inductive':
            pipeline.run_inductive_generation(args)
        elif args.mode == 'deductive':
            if not args.codebook:
                parser.error("--codebook required for deductive mode")
            pipeline.run_deductive_coding(args)
        elif args.mode == 'iterative':
            pipeline.run_iterative_refinement(args)
        elif args.mode == 'validate':
            if not args.human_codes or not args.ai_codes:
                parser.error("--human-codes and --ai-codes required for validation")
            pipeline.run_validation(args)
        elif args.mode == 'cue-response':
            pipeline.run_cue_response_analysis(args)
            
    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
