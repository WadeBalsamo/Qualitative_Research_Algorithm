#!/usr/bin/env python3
"""
guided_analysis.py - Guided interface for clinical researchers
Provides a user-friendly walkthrough of the MORE analysis pipeline
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

# Import our modules
from generate_codebook import LlamaCodebookGenerator
from clinical_analysis_tools import ClinicalAnalysisTools

class GuidedAnalysis:
    """
    Interactive guided analysis for clinical researchers
    """
    
    def __init__(self):
        self.generator = None
        self.clinical_tools = None
        self.transcripts = None
        self.codebook = None
        
    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
        
    def print_option(self, number: str, text: str):
        """Print formatted option"""
        print(f"\n  [{number}] {text}")
        
    def get_choice(self, prompt: str, valid_options: list) -> str:
        """Get validated user choice"""
        while True:
            choice = input(f"\n{prompt}: ").strip()
            if choice in valid_options:
                return choice
            print(f"Invalid choice. Please enter one of: {', '.join(valid_options)}")
    
    def start(self):
        """Start the guided analysis"""
        self.print_header("WELCOME TO MORE QUALITATIVE ANALYSIS")
        print("\nThis guided interface will help you analyze your")
        print("Mindfulness-Oriented Recovery Enhancement session transcripts.")
        
        input("\nPress Enter to begin...")
        
        # Check system requirements
        self.check_requirements()
        
        # Main menu loop
        while True:
            self.show_main_menu()
    
    def check_requirements(self):
        """Check if system is properly set up"""
        self.print_header("SYSTEM CHECK")
        
        print("\nChecking requirements...")
        
        # Check for transcript directory
        if not os.path.exists("more_transcripts"):
            print("⚠️  Transcript directory not found!")
            create = self.get_choice("Create directory now? (y/n)", ["y", "n"])
            if create == "y":
                os.makedirs("more_transcripts")
                print("✓ Created 'more_transcripts' directory")
                print("  Please place your .srt files there")
                input("\nPress Enter when ready...")
            else:
                print("Please create 'more_transcripts' directory and add files")
                sys.exit(0)
        else:
            srt_files = list(Path("more_transcripts").glob("*.srt"))
            print(f"✓ Found {len(srt_files)} transcript files")
        
        # Check for output directories
        for dir_name in ["codebooks", "analysis_output", "clinical_reports"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        print("✓ Output directories ready")
        
        # Initialize components
        print("\nInitializing analysis components...")
        print("  (This may take a few minutes on first run)")
        
        try:
            self.generator = LlamaCodebookGenerator(
                clinical_bert=True,
                quantization=True
            )
            self.clinical_tools = ClinicalAnalysisTools(self.generator)
            print("✓ Analysis components initialized")
        except Exception as e:
            print(f"❌ Error initializing: {e}")
            print("\nPlease ensure you have:")
            print("  1. Sufficient RAM (256GB+ recommended)")
            print("  2. CUDA-capable GPU (optional but recommended)")
            print("  3. All required Python packages installed")
            sys.exit(1)
    
    def show_main_menu(self):
        """Display main menu"""
        self.print_header("MAIN MENU")
        
        self.print_option("1", "Generate NEW codebook from transcripts")
        self.print_option("2", "Apply EXISTING codebook")
        self.print_option("3", "Analyze therapeutic progression")
        self.print_option("4", "Validate coding (compare with human)")
        self.print_option("5", "View quick statistics")
        self.print_option("Q", "Quit")
        
        choice = self.get_choice("Select an option", ["1", "2", "3", "4", "5", "q", "Q"])
        
        if choice == "1":
            self.generate_new_codebook()
        elif choice == "2":
            self.apply_existing_codebook()
        elif choice == "3":
            self.analyze_progression()
        elif choice == "4":
            self.validate_coding()
        elif choice == "5":
            self.show_statistics()
        elif choice.lower() == "q":
            print("\nThank you for using MORE Analysis!")
            sys.exit(0)
    
    def generate_new_codebook(self):
        """Generate new codebook workflow"""
        self.print_header("GENERATE NEW CODEBOOK")
        
        # Load transcripts
        print("\nLoading transcripts...")
        self.transcripts = self.generator.load_transcripts("more_transcripts")
        print(f"✓ Loaded {len(self.transcripts)} sessions")
        
        # Ask about approach
        print("\nHow would you like to generate the codebook?")
        self.print_option("1", "Fully automated (faster)")
        self.print_option("2", "Interactive refinement (recommended)")
        
        approach = self.get_choice("Select approach", ["1", "2"])
        interactive = (approach == "2")
        
        # Ask about number of codes
        print("\nHow many codes would you like to generate?")
        print("  Recommended: 15-25 for initial analysis")
        
        while True:
            try:
                n_codes = int(input("Number of codes (default: 20): ") or "20")
                if 5 <= n_codes <= 50:
                    break
                print("Please enter a number between 5 and 50")
            except ValueError:
                print("Please enter a valid number")
        
        # Generate codebook
        print(f"\nGenerating {n_codes} codes...")
        print("This will take approximately 5-10 minutes...")
        
        self.codebook = self.generator.generate_inductive_codebook(
            self.transcripts,
            n_codes=n_codes,
            interactive_refinement=interactive
        )
        
        # Save codebook
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        codebook_path = f"codebooks/generated_{timestamp}.json"
        self.generator.save_codebook(codebook_path)
        
        print(f"\n✓ Codebook saved to: {codebook_path}")
        
        # Ask about next steps
        print("\nWould you like to:")
        self.print_option("1", "Apply this codebook to transcripts")
        self.print_option("2", "Generate clinical report")
        self.print_option("3", "Return to main menu")
        
        next_step = self.get_choice("Select option", ["1", "2", "3"])
        
        if next_step == "1":
            self.apply_codebook_to_transcripts()
        elif next_step == "2":
            self.generate_report()
        
    def apply_existing_codebook(self):
        """Apply existing codebook workflow"""
        self.print_header("APPLY EXISTING CODEBOOK")
        
        # List available codebooks
        codebook_files = list(Path("codebooks").glob("*.json"))
        
        if not codebook_files:
            print("\n⚠️  No codebooks found!")
            print("Please generate a codebook first")
            input("\nPress Enter to return to main menu...")
            return
        
        print("\nAvailable codebooks:")
        for i, file in enumerate(codebook_files, 1):
            print(f"  [{i}] {file.name}")
        
        # Select codebook
        while True:
            try:
                selection = int(input(f"\nSelect codebook (1-{len(codebook_files)}): "))
                if 1 <= selection <= len(codebook_files):
                    break
            except ValueError:
                pass
            print("Please enter a valid number")
        
        # Load codebook
        codebook_path = str(codebook_files[selection - 1])
        self.generator.load_codebook(codebook_path)
        print(f"✓ Loaded codebook with {len(self.generator.codebook)} codes")
        
        # Load transcripts if needed
        if not self.transcripts:
            print("\nLoading transcripts...")
            self.transcripts = self.generator.load_transcripts("more_transcripts")
        
        # Apply codebook
        self.apply_codebook_to_transcripts()
    
    def apply_codebook_to_transcripts(self):
        """Apply loaded codebook to transcripts"""
        print("\nApplying codebook to transcripts...")
        print("This will take approximately 2-5 minutes...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"analysis_output/coded_{timestamp}"
        
        coded_results = self.generator.apply_codebook_to_transcripts(
            self.transcripts,
            output_dir=output_dir
        )
        
        print(f"\n✓ Coding complete!")
        print(f"  Output saved to: {output_dir}")
        
        # Show summary statistics
        total_codes = sum(len(r['codes_applied']) for r in coded_results)
        print(f"\n📊 Summary Statistics:")
        print(f"  • Total segments coded: {total_codes}")
        print(f"  • Average per session: {total_codes/len(coded_results):.1f}")
        
        # Save results for later analysis
        self.coded_results = coded_results
        
        # Ask about report
        generate = self.get_choice("\nGenerate clinical report? (y/n)", ["y", "n"])
        if generate == "y":
            self.generate_report()
    
    def analyze_progression(self):
        """Analyze therapeutic progression"""
        self.print_header("ANALYZE THERAPEUTIC PROGRESSION")
        
        # Check if we have coded results
        if not hasattr(self, 'coded_results'):
            print("\n⚠️  No coded results found!")
            print("Please apply a codebook first")
            input("\nPress Enter to return to main menu...")
            return
        
        print("\nAnalyzing progression across sessions...")
        
        # Perform analysis
        progression_df = self.clinical_tools.analyze_therapeutic_progression(
            self.coded_results
        )
        
        # Show key findings
        patterns = self.clinical_tools.analysis_results.get('progression', {}).get('patterns', {})
        
        print("\n📈 KEY FINDINGS:")
        
        if patterns.get('increasing'):
            print("\n✓ Codes showing INCREASE over time (therapeutic gains):")
            for code, slope, p_value in patterns['increasing'][:5]:
                print(f"  • {code}: {slope:.3f} per session (p={p_value:.3f})")
        
        if patterns.get('decreasing'):
            print("\n✓ Codes showing DECREASE over time (symptom reduction):")
            for code, slope, p_value in patterns['decreasing'][:5]:
                print(f"  • {code}: {abs(slope):.3f} per session (p={p_value:.3f})")
        
        if patterns.get('stable'):
            print(f"\n✓ Stable patterns: {len(patterns['stable'])} codes")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"clinical_reports/progression_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump({
                'patterns': patterns,
                'data': progression_df.to_dict()
            }, f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {results_path}")
        input("\nPress Enter to continue...")
    
    def validate_coding(self):
        """Validate against human coding"""
        self.print_header("VALIDATE CODING")
        
        print("\nThis function compares AI coding with human coding")
        print("to assess inter-rater reliability.")
        
        # Check for files
        print("\nPlease ensure you have:")
        print("  1. Human coded file (JSON format)")
        print("  2. AI coded file (JSON format)")
        print("\nBoth should have the same segment IDs")
        
        # Get file paths
        human_file = input("\nPath to human coded file: ").strip()
        ai_file = input("Path to AI coded file: ").strip()
        
        # Validate files exist
        if not os.path.exists(human_file):
            print(f"❌ File not found: {human_file}")
            input("\nPress Enter to return...")
            return
        
        if not os.path.exists(ai_file):
            print(f"❌ File not found: {ai_file}")
            input("\nPress Enter to return...")
            return
        
        # Run validation
        print("\nCalculating reliability metrics...")
        
        try:
            reliability = self.clinical_tools.validate_inter_rater_reliability(
                human_file, ai_file
            )
            
            # Display results
            print("\n📊 VALIDATION RESULTS:")
            print(f"\n  Cohen's Kappa: {reliability['cohen_kappa']:.3f}")
            print(f"  Percent Agreement: {reliability['percent_agreement']:.1%}")
            print(f"  Segments Compared: {reliability['n_segments']}")
            
            # Interpretation
            kappa = reliability['cohen_kappa']
            if kappa >= 0.81:
                interpretation = "Almost perfect agreement! 🎉"
            elif kappa >= 0.61:
                interpretation = "Substantial agreement 👍"
            elif kappa >= 0.41:
                interpretation = "Moderate agreement"
            elif kappa >= 0.21:
                interpretation = "Fair agreement"
            else:
                interpretation = "Poor agreement - review needed"
            
            print(f"\n  Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"\n❌ Error during validation: {e}")
            print("Please check file formats")
        
        input("\nPress Enter to continue...")
    
    def show_statistics(self):
        """Show quick statistics"""
        self.print_header("QUICK STATISTICS")
        
        # Load transcript info
        if not self.transcripts:
            self.transcripts = self.generator.load_transcripts("more_transcripts")
        
        print(f"\n📁 Transcript Statistics:")
        print(f"  • Total sessions: {len(self.transcripts)}")
        
        total_subtitles = sum(len(subs) for subs in self.transcripts.values())
        print(f"  • Total segments: {total_subtitles}")
        
        # Session lengths
        session_lengths = {
            sid: sum(sub.end.total_seconds() - sub.start.total_seconds() 
                    for sub in subs)
            for sid, subs in self.transcripts.items()
        }
        
        avg_length = sum(session_lengths.values()) / len(session_lengths) / 60
        print(f"  • Average session length: {avg_length:.1f} minutes")
        
        # Codebook info
        codebook_files = list(Path("codebooks").glob("*.json"))
        print(f"\n📚 Codebook Statistics:")
        print(f"  • Available codebooks: {len(codebook_files)}")
        
        # Coded output info
        output_dirs = list(Path("analysis_output").glob("coded_*"))
        print(f"\n📊 Analysis Statistics:")
        print(f"  • Coding runs completed: {len(output_dirs)}")
        
        # Report info
        report_files = list(Path("clinical_reports").glob("*.pdf"))
        print(f"  • Clinical reports generated: {len(report_files)}")
        
        input("\nPress Enter to continue...")
    
    def generate_report(self):
        """Generate clinical report"""
        print("\nGenerating comprehensive clinical report...")
        print("This will take approximately 2-3 minutes...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"clinical_reports/clinical_report_{timestamp}.pdf"
        
        # Ensure we have necessary data
        if not hasattr(self, 'coded_results'):
            print("⚠️  Coded results not found. Applying codebook first...")
            self.apply_codebook_to_transcripts()
        
        # Generate report
        self.clinical_tools.generate_clinical_report(
            self.generator.codebook,
            self.coded_results,
            report_path
        )
        
        print(f"\n✓ Report generated: {report_path}")
        
        # Also export codebook
        print("\nExporting codebook in clinical formats...")
        
        excel_path = self.clinical_tools.export_for_clinical_use(
            self.generator.codebook, 'excel'
        )
        print(f"  • Excel: {excel_path}")
        
        redcap_path = self.clinical_tools.export_for_clinical_use(
            self.generator.codebook, 'redcap'
        )
        print(f"  • REDCap: {redcap_path}")
        
        input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    analysis = GuidedAnalysis()
    
    try:
        analysis.start()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        print("Thank you for using MORE Analysis!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nPlease contact support with this error message")
        sys.exit(1)


if __name__ == "__main__":
    main()
