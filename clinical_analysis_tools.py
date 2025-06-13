#!/usr/bin/env python3
"""
clinical_analysis_tools.py - Clinical Research Analysis Tools for MORE Study
Provides visualization, validation, and export capabilities for qualitative coding
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
from pathlib import Path
import logging

# Statistical analysis
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import the main generator
from generate_codebook import LlamaCodebookGenerator

logger = logging.getLogger(__name__)

class ClinicalAnalysisTools:
    """
    Analysis tools for clinical researchers working with MORE qualitative data
    """
    
    def __init__(self, codebook_generator: Optional[LlamaCodebookGenerator] = None):
        """
        Initialize analysis tools
        
        Args:
            codebook_generator: Instance of LlamaCodebookGenerator
        """
        self.generator = codebook_generator or LlamaCodebookGenerator()
        self.analysis_results = {}
        
        # MORE-specific outcome domains
        self.outcome_domains = {
            'pain_severity': ['pain intensity', 'pain relief', 'pain management'],
            'disability': ['functional limitation', 'physical impairment', 'activity restriction'],
            'quality_of_life': ['well-being', 'life satisfaction', 'positive affect'],
            'mindfulness': ['present moment', 'non-judgment', 'awareness'],
            'catastrophizing': ['rumination', 'magnification', 'helplessness']
        }
    
    def validate_inter_rater_reliability(self, 
                                       human_coded_file: str,
                                       ai_coded_file: str) -> Dict:
        """
        Calculate inter-rater reliability between human and AI coding
        
        Returns:
            Dictionary with reliability metrics
        """
        # Load coded data
        human_codes = self._load_coded_data(human_coded_file)
        ai_codes = self._load_coded_data(ai_coded_file)
        
        # Align segments
        aligned_human = []
        aligned_ai = []
        
        for segment_id in human_codes.keys():
            if segment_id in ai_codes:
                aligned_human.append(human_codes[segment_id])
                aligned_ai.append(ai_codes[segment_id])
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(aligned_human, aligned_ai)
        
        # Calculate percent agreement
        agreement = np.mean([h == a for h, a in zip(aligned_human, aligned_ai)])
        
        # Create confusion matrix
        cm = confusion_matrix(aligned_human, aligned_ai)
        
        # Category-specific agreement
        category_agreement = {}
        unique_codes = list(set(aligned_human + aligned_ai))
        
        for code in unique_codes:
            human_mask = [h == code for h in aligned_human]
            ai_mask = [a == code for a in aligned_ai]
            category_agreement[code] = np.mean([h == a for h, a in zip(human_mask, ai_mask)])
        
        results = {
            'cohen_kappa': kappa,
            'percent_agreement': agreement,
            'n_segments': len(aligned_human),
            'confusion_matrix': cm,
            'category_agreement': category_agreement
        }
        
        self.analysis_results['inter_rater'] = results
        return results
    
    def analyze_therapeutic_progression(self, coded_sessions: List[Dict]) -> pd.DataFrame:
        """
        Analyze progression of therapeutic themes across sessions
        
        Args:
            coded_sessions: List of coded session data
            
        Returns:
            DataFrame with progression analysis
        """
        # Create timeline of code occurrences
        progression_data = []
        
        for session_data in coded_sessions:
            session_num = int(session_data['session_id'].split('_')[1])
            
            code_counts = pd.Series(session_data['codes_applied']).value_counts()
            
            for code, count in code_counts.items():
                progression_data.append({
                    'session': session_num,
                    'code': code,
                    'count': count,
                    'normalized_count': count / len(session_data['codes_applied'])
                })
        
        df = pd.DataFrame(progression_data)
        
        # Identify progression patterns
        patterns = self._identify_progression_patterns(df)
        
        self.analysis_results['progression'] = {
            'data': df,
            'patterns': patterns
        }
        
        return df
    
    def _identify_progression_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify patterns in code progression"""
        patterns = {
            'increasing': [],
            'decreasing': [],
            'stable': [],
            'u_shaped': [],
            'inverted_u': []
        }
        
        for code in df['code'].unique():
            code_data = df[df['code'] == code].groupby('session')['normalized_count'].mean()
            
            if len(code_data) < 3:
                continue
            
            # Fit linear regression
            x = code_data.index.values
            y = code_data.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Classify pattern
            if p_value < 0.05:
                if slope > 0:
                    patterns['increasing'].append((code, slope, p_value))
                else:
                    patterns['decreasing'].append((code, slope, p_value))
            else:
                # Check for non-linear patterns
                mid_point = len(y) // 2
                first_half_mean = np.mean(y[:mid_point])
                second_half_mean = np.mean(y[mid_point:])
                peak = np.max(y)
                trough = np.min(y)
                
                if peak == y[mid_point] and peak > first_half_mean * 1.2:
                    patterns['inverted_u'].append(code)
                elif trough == y[mid_point] and trough < first_half_mean * 0.8:
                    patterns['u_shaped'].append(code)
                else:
                    patterns['stable'].append(code)
        
        return patterns
    
    def generate_clinical_report(self, 
                               codebook: List[Dict],
                               coded_sessions: List[Dict],
                               output_file: str = "clinical_analysis_report.pdf") -> None:
        """
        Generate comprehensive clinical analysis report
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_file) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'MORE Qualitative Analysis Report', 
                    ha='center', size=24, weight='bold')
            fig.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d")}', 
                    ha='center', size=14)
            fig.text(0.5, 0.5, f'Sessions Analyzed: {len(coded_sessions)}', 
                    ha='center', size=12)
            fig.text(0.5, 0.4, f'Codes Applied: {len(codebook)}', 
                    ha='center', size=12)
            pdf.savefig(fig)
            plt.close()
            
            # Code frequency analysis
            self._plot_code_frequencies(coded_sessions, pdf)
            
            # Therapeutic progression
            self._plot_progression_analysis(coded_sessions, pdf)
            
            # Network analysis
            self._plot_code_network(codebook, coded_sessions, pdf)
            
            # Word clouds by domain
            self._plot_domain_wordclouds(codebook, coded_sessions, pdf)
            
            # Clinical insights summary
            self._plot_clinical_insights(codebook, coded_sessions, pdf)
        
        logger.info(f"Clinical report saved to {output_file}")
    
    def _plot_code_frequencies(self, coded_sessions: List[Dict], pdf):
        """Plot code frequency analysis"""
        # Aggregate code frequencies
        all_codes = []
        for session in coded_sessions:
            all_codes.extend(session['codes_applied'])
        
        code_counts = pd.Series(all_codes).value_counts().head(20)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        code_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Most Frequent Codes', fontsize=16, weight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _plot_progression_analysis(self, coded_sessions: List[Dict], pdf):
        """Plot therapeutic progression analysis"""
        df = self.analyze_therapeutic_progression(coded_sessions)
        
        # Select key codes for visualization
        key_codes = ['Pain Relief', 'Metacognitive Awareness', 'Experiential Avoidance', 
                     'Attention Regulation', 'Reappraisal']
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, code in enumerate(key_codes):
            if i >= len(axes) - 1:
                break
                
            code_data = df[df['code'] == code]
            if not code_data.empty:
                session_means = code_data.groupby('session')['normalized_count'].mean()
                axes[i].plot(session_means.index, session_means.values, 
                           marker='o', linewidth=2, markersize=8)
                axes[i].set_title(code)
                axes[i].set_xlabel('Session')
                axes[i].set_ylabel('Normalized Frequency')
                axes[i].grid(alpha=0.3)
        
        # Hide empty subplot
        axes[-1].axis('off')
        
        plt.suptitle('Therapeutic Construct Progression Across Sessions', 
                    fontsize=16, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _plot_code_network(self, codebook: List[Dict], coded_sessions: List[Dict], pdf):
        """Plot network analysis of code co-occurrences"""
        # Build co-occurrence matrix
        code_names = [code['category'] for code in codebook]
        co_occurrence = np.zeros((len(code_names), len(code_names)))
        
        for session in coded_sessions:
            codes_in_session = list(set(session['codes_applied']))
            for i, code1 in enumerate(code_names):
                for j, code2 in enumerate(code_names):
                    if i != j and code1 in codes_in_session and code2 in codes_in_session:
                        co_occurrence[i, j] += 1
        
        # Create network
        G = nx.Graph()
        threshold = np.percentile(co_occurrence[co_occurrence > 0], 75)
        
        for i, code1 in enumerate(code_names):
            for j, code2 in enumerate(code_names):
                if i < j and co_occurrence[i, j] > threshold:
                    G.add_edge(code1, code2, weight=co_occurrence[i, j])
        
        # Plot network
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node sizes based on degree
        node_sizes = [G.degree(node) * 300 for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)
        
        ax.set_title('Code Co-occurrence Network', fontsize=16, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _plot_domain_wordclouds(self, codebook: List[Dict], coded_sessions: List[Dict], pdf):
        """Generate word clouds for theoretical domains"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        domains = ['attention_regulation', 'pain_experience', 'metacognitive_awareness', 'reappraisal']
        
        for i, domain in enumerate(domains):
            # Collect text for this domain
            domain_text = []
            for code in codebook:
                if domain in str(code.get('theoretical_domain', '')):
                    domain_text.extend(code['codes'])
                    domain_text.append(code['category'])
            
            if domain_text:
                # Create word cloud
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white').generate(' '.join(domain_text))
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(domain.replace('_', ' ').title())
                axes[i].axis('off')
        
        plt.suptitle('Key Concepts by Theoretical Domain', fontsize=16, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _plot_clinical_insights(self, codebook: List[Dict], coded_sessions: List[Dict], pdf):
        """Generate clinical insights summary"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Calculate key metrics
        total_segments = sum(len(s['codes_applied']) for s in coded_sessions)
        unique_codes = len(set(code for s in coded_sessions for code in s['codes_applied']))
        avg_codes_per_session = np.mean([len(s['codes_applied']) for s in coded_sessions])
        
        # Identify most improved domains
        progression_df = self.analyze_therapeutic_progression(coded_sessions)
        patterns = self.analysis_results.get('progression', {}).get('patterns', {})
        
        # Create insights text
        insights = f"""
        CLINICAL INSIGHTS SUMMARY
        
        Dataset Overview:
        • Total coded segments: {total_segments:,}
        • Unique codes applied: {unique_codes}
        • Average codes per session: {avg_codes_per_session:.1f}
        
        Key Therapeutic Progressions:
        """
        
        if patterns.get('increasing'):
            insights += "\n\nIncreasing Patterns (therapeutic gains):\n"
            for code, slope, p_value in patterns['increasing'][:5]:
                insights += f"  • {code}: {slope:.3f} increase per session (p={p_value:.3f})\n"
        
        if patterns.get('decreasing'):
            insights += "\n\nDecreasing Patterns (symptom reduction):\n"
            for code, slope, p_value in patterns['decreasing'][:5]:
                insights += f"  • {code}: {abs(slope):.3f} decrease per session (p={p_value:.3f})\n"
        
        insights += """
        
        Clinical Recommendations:
        1. Focus on reinforcing codes showing positive progression
        2. Address persistent challenges identified in stable patterns
        3. Consider session-specific adaptations based on progression data
        4. Monitor inverted-U patterns for optimal intervention timing
        """
        
        ax.text(0.1, 0.9, insights, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.title('Clinical Insights and Recommendations', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def export_for_clinical_use(self, 
                               codebook: List[Dict],
                               output_format: str = 'excel') -> str:
        """
        Export codebook in clinical-friendly format
        
        Args:
            codebook: Generated codebook
            output_format: 'excel', 'csv', or 'redcap'
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'excel':
            filename = f'MORE_Codebook_{timestamp}.xlsx'
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main codebook sheet
                codebook_df = pd.DataFrame([{
                    'Code_ID': i+1,
                    'Category': code['category'],
                    'Description': code['description'],
                    'Keywords': ', '.join(code['codes']),
                    'Include': code['inclusive'],
                    'Exclude': code['exclusive'],
                    'Domain': code.get('theoretical_domain', 'Unspecified'),
                    'Mean_Similarity': code.get('score_metrics', {}).get('mean_similarity', 0),
                    'Coverage': code.get('score_metrics', {}).get('coverage', 0)
                } for i, code in enumerate(codebook)])
                
                codebook_df.to_excel(writer, sheet_name='Codebook', index=False)
                
                # Theoretical domains sheet
                domains_df = pd.DataFrame([
                    {'Domain': domain, 'Keywords': ', '.join(keywords)}
                    for domain, keywords in self.generator.theoretical_domains.items()
                ])
                domains_df.to_excel(writer, sheet_name='Theoretical_Domains', index=False)
                
                # Formatting
                workbook = writer.book
                for sheet in workbook.worksheets:
                    for column in sheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        sheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        elif output_format == 'csv':
            filename = f'MORE_Codebook_{timestamp}.csv'
            codebook_df = pd.DataFrame(codebook)
            codebook_df.to_csv(filename, index=False)
        
        elif output_format == 'redcap':
            filename = f'MORE_Codebook_REDCap_{timestamp}.csv'
            # Format for REDCap data dictionary
            redcap_df = pd.DataFrame([{
                'Variable / Field Name': f"code_{i+1}",
                'Form Name': 'more_qualitative_coding',
                'Section Header': code.get('theoretical_domain', 'General'),
                'Field Type': 'checkbox',
                'Field Label': code['category'],
                'Choices, Calculations, OR Slider Labels': '1, Present | 0, Absent',
                'Field Note': f"{code['description']}. Include: {code['inclusive']}. Exclude: {code['exclusive']}",
                'Required Field?': 'y',
                'Identifier?': '',
                'Branching Logic': '',
                'Custom Alignment': '',
                'Question Number': ''
            } for i, code in enumerate(codebook)])
            
            redcap_df.to_csv(filename, index=False)
        
        logger.info(f"Codebook exported to {filename}")
        return filename
    
    def _load_coded_data(self, filepath: str) -> Dict:
        """Load coded data from file"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {filepath}")


def demonstrate_clinical_workflow():
    """
    Demonstrate complete clinical research workflow
    """
    print("=== MORE CLINICAL ANALYSIS WORKFLOW ===\n")
    
    # Initialize tools
    generator = LlamaCodebookGenerator(clinical_bert=True)
    clinical_tools = ClinicalAnalysisTools(generator)
    
    # Load transcripts
    transcript_dir = "more_transcripts"
    transcripts = generator.load_transcripts(transcript_dir)
    
    # Generate codebook with clinical focus
    print("1. Generating clinically-informed codebook...")
    codebook = generator.generate_inductive_codebook(
        transcripts,
        n_codes=25,
        interactive_refinement=True
    )
    
    # Apply codebook
    print("\n2. Applying codebook to transcripts...")
    coded_results = generator.apply_codebook_to_transcripts(transcripts)
    
    # Validate if human coding available
    if os.path.exists("human_coded_sample.json"):
        print("\n3. Validating against human coding...")
        reliability = clinical_tools.validate_inter_rater_reliability(
            "human_coded_sample.json",
            "ai_coded_sample.json"
        )
        print(f"   Cohen's Kappa: {reliability['cohen_kappa']:.3f}")
        print(f"   Percent Agreement: {reliability['percent_agreement']:.1%}")
    
    # Analyze progression
    print("\n4. Analyzing therapeutic progression...")
    progression_df = clinical_tools.analyze_therapeutic_progression(coded_results)
    
    # Generate clinical report
    print("\n5. Generating clinical analysis report...")
    clinical_tools.generate_clinical_report(codebook, coded_results)
    
    # Export for clinical use
    print("\n6. Exporting codebook for clinical use...")
    excel_file = clinical_tools.export_for_clinical_use(codebook, 'excel')
    redcap_file = clinical_tools.export_for_clinical_use(codebook, 'redcap')
    
    print(f"\nExported files:")
    print(f"  - Excel: {excel_file}")
    print(f"  - REDCap: {redcap_file}")
    
    print("\n=== WORKFLOW COMPLETE ===")


if __name__ == "__main__":
    demonstrate_clinical_workflow()
