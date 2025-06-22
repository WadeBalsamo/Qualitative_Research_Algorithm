#!/usr/bin/env python3
"""
generate_codebook.py - Automated Qualitative Codebook Generation using Llama3-70B
Implements iterative inductive-deductive-inductive approach for MORE study analysis
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict, Counter
import re

# NLP and ML imports
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import srt
from tqdm import tqdm

# Local imports (assuming apply_codebook.py is in same directory)
from apply_codebook import apply_research_codes_to_sentences

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaCodebookGenerator:
    """
    Generates inductive codebooks using Llama3-70B for qualitative analysis
    of Mindfulness-Oriented Recovery Enhancement (MORE) sessions
    """
    
    def __init__(self, 
                 model_path: str = "meta-llama/Llama-3-70b-chat-hf",
                 bert_model: str = "bert-base-uncased",
                 clinical_bert: bool = True,
                 quantization: bool = True):
        """
        Initialize the codebook generator with Llama3 and BERT models
        
        Args:
            model_path: Path to Llama3-70B model
            bert_model: BERT model for embeddings (or ClinicalBERT)
            clinical_bert: Use ClinicalBERT variant if True
            quantization: Use 4-bit quantization for memory efficiency
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_llama_model(quantization)
        self._init_bert_model(bert_model, clinical_bert)
        
        # Initialize codebook storage
        self.codebook = []
        self.codebook_history = []
        self.participant_summaries = {}
        self.instructor_summaries = {}
        
        # MORE-specific theoretical domains
        self.theoretical_domains = {
            "attention_regulation": ["focus", "concentration", "awareness", "distraction"],
            "pain_experience": ["pain", "discomfort", "sensation", "relief"],
            "avoidance_patterns": ["escape", "avoid", "fear", "resistance"],
            "metacognitive_awareness": ["noticing", "observing", "awareness of awareness"],
            "reappraisal": ["reframe", "perspective", "acceptance", "non-judgment"],
            "savoring": ["pleasure", "enjoyment", "appreciation", "positive"],
            "somatic_awareness": ["body", "physical", "sensation", "breath"],
            "therapeutic_engagement": ["motivation", "practice", "commitment", "goals"]
        }
        
    def _init_llama_model(self, quantization: bool):
        """Initialize Llama3-70B with optional quantization"""
        logger.info("Initializing Llama3-70B model...")
        
        if quantization:
            # 4-bit quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
    def _init_bert_model(self, bert_model: str, clinical_bert: bool):
        """Initialize BERT or ClinicalBERT for embeddings"""
        if clinical_bert:
            # Use ClinicalBERT for medical domain specificity
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
        else:
            model_name = bert_model
            
        logger.info(f"Initializing {model_name} for embeddings...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def load_transcripts(self, transcript_dir: str) -> Dict[str, List[srt.Subtitle]]:
        """
        Load SRT transcripts from directory
        
        Returns:
            Dictionary mapping session IDs to subtitle lists
        """
        transcripts = {}
        
        for file_path in Path(transcript_dir).glob("*.srt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                subtitles = list(srt.parse(f.read()))
                session_id = file_path.stem
                transcripts[session_id] = subtitles
                
        logger.info(f"Loaded {len(transcripts)} transcript files")
        return transcripts
    
    def separate_speakers(self, subtitles: List[srt.Subtitle]) -> Tuple[List[str], List[str]]:
        """
        Separate instructor and participant dialogue
        
        Returns:
            Tuple of (instructor_texts, participant_texts)
        """
        instructor_texts = []
        participant_texts = []
        
        for sub in subtitles:
            if "instructor:" in sub.content.lower() or "main_speaker:" in sub.content.lower():
                instructor_texts.append(sub.content)
            else:
                participant_texts.append(sub.content)
                
        return instructor_texts, participant_texts
    
    def generate_session_summary(self, texts: List[str], speaker_type: str) -> str:
        """
        Generate summary of session using Llama3-70B
        
        Args:
            texts: List of utterances
            speaker_type: "instructor" or "participant"
            
        Returns:
            Generated summary
        """
        # Combine texts
        combined_text = " ".join(texts)[:4000]  # Limit context length
        
        # Create prompt based on speaker type
        if speaker_type == "instructor":
            prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are analyzing a Mindfulness-Oriented Recovery Enhancement (MORE) session.
            Summarize the key instructional cues and teachings from the instructor.
            Focus on: meditation instructions, pain management techniques, attention training,
            reappraisal strategies, and therapeutic guidance.
            <|eot_id|>
            
            <|start_header_id|>user<|end_header_id|>
            Instructor dialogue:
            {combined_text}
            
            Provide a concise summary of the main instructional elements:
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            """
        else:
            prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are analyzing participant responses in a Mindfulness-Oriented Recovery Enhancement session.
            Summarize the key experiences, challenges, and insights shared by participants.
            Focus on: pain experiences, mindfulness practice difficulties, breakthroughs,
            emotional responses, and engagement with techniques.
            <|eot_id|>
            
            <|start_header_id|>user<|end_header_id|>
            Participant dialogue:
            {combined_text}
            
            Provide a concise summary of participant experiences:
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            """
        
        # Generate summary
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )
        
        summary = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        summary = summary.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return summary
    
    def extract_key_concepts(self, summary: str, num_concepts: int = 10) -> List[str]:
        """
        Extract key concepts from summary using Llama3
        
        Returns:
            List of key concepts/themes
        """
        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Extract the {num_concepts} most important concepts, themes, or patterns from this summary.
        Focus on clinically relevant aspects of mindfulness practice and pain management.
        Return only the list of concepts, one per line.
        <|eot_id|>
        
        <|start_header_id|>user<|end_header_id|>
        Summary:
        {summary}
        <|eot_id|>
        
        <|start_header_id|>assistant<|end_header_id|>
        Key concepts:
        """
        
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )
        
        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        concepts = response.split("Key concepts:")[-1].strip().split("\n")
        concepts = [c.strip("- ").strip() for c in concepts if c.strip()][:num_concepts]
        
        return concepts
    
    def cluster_concepts(self, all_concepts: List[str], n_clusters: int = 15) -> Dict[int, List[str]]:
        """
        Cluster similar concepts using embeddings
        
        Returns:
            Dictionary mapping cluster IDs to concept lists
        """
        # Get embeddings for all concepts
        embeddings = []
        
        for concept in tqdm(all_concepts, desc="Embedding concepts"):
            inputs = self.bert_tokenizer(concept, return_tensors="pt", 
                                       truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        embeddings = np.array(embeddings)
        
        # Cluster using agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group concepts by cluster
        clusters = defaultdict(list)
        for concept, label in zip(all_concepts, cluster_labels):
            clusters[label].append(concept)
            
        return dict(clusters)
    
    def generate_code_definition(self, concepts: List[str], theoretical_domain: Optional[str] = None) -> Dict:
        """
        Generate a code definition from clustered concepts
        
        Returns:
            Dictionary with code structure matching apply_codebook.py format
        """
        concepts_text = "\n".join(f"- {c}" for c in concepts[:10])  # Limit to top 10
        
        domain_context = ""
        if theoretical_domain:
            domain_context = f"This code should relate to the theoretical domain of {theoretical_domain}."
        
        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are creating a qualitative research code for a Mindfulness-Oriented Recovery Enhancement study.
        Based on the provided concepts, create a code definition with:
        1. A concise category name (2-4 words)
        2. A brief description (1-2 sentences)
        3. 3-5 specific code keywords
        4. Inclusive criteria (what should be coded)
        5. Exclusive criteria (what should NOT be coded)
        {domain_context}
        <|eot_id|>
        
        <|start_header_id|>user<|end_header_id|>
        Concepts from participant data:
        {concepts_text}
        
        Create a code definition in JSON format.
        <|eot_id|>
        
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )
        
        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                code_def = json.loads(json_match.group())
            else:
                # Fallback: create structured dict from response
                code_def = self._parse_code_response(response, concepts)
        except:
            # Fallback if JSON parsing fails
            code_def = self._parse_code_response(response, concepts)
            
        return code_def
    
    def _parse_code_response(self, response: str, concepts: List[str]) -> Dict:
        """Fallback parser for code definitions"""
        lines = response.split('\n')
        
        code_def = {
            'category': 'Emergent Theme',
            'description': 'Participant-reported experience',
            'codes': concepts[:3],
            'inclusive': ' '.join(concepts[:5]),
            'exclusive': 'Unrelated to mindfulness practice or pain experience'
        }
        
        # Try to extract structured information
        for line in lines:
            if 'category' in line.lower() or 'name' in line.lower():
                code_def['category'] = line.split(':')[-1].strip().strip('"\'')
            elif 'description' in line.lower():
                code_def['description'] = line.split(':')[-1].strip().strip('"\'')
            elif 'inclusive' in line.lower():
                code_def['inclusive'] = line.split(':')[-1].strip().strip('"\'')
            elif 'exclusive' in line.lower():
                code_def['exclusive'] = line.split(':')[-1].strip().strip('"\'')
                
        return code_def
    
    def refine_codebook_with_researcher(self, codebook: List[Dict]) -> List[Dict]:
        """
        Interactive refinement process with researcher
        
        Returns:
            Refined codebook based on researcher input
        """
        refined_codebook = []
        
        print("\n" + "="*50)
        print("CODEBOOK REFINEMENT PROCESS")
        print("="*50)
        
        for i, code in enumerate(codebook):
            print(f"\nCode {i+1}/{len(codebook)}:")
            print(f"Category: {code['category']}")
            print(f"Description: {code['description']}")
            print(f"Keywords: {', '.join(code['codes'])}")
            print(f"Include: {code['inclusive']}")
            print(f"Exclude: {code['exclusive']}")
            
            action = input("\nAction? (k)eep, (m)odify, (d)elete, (s)kip: ").lower()
            
            if action == 'k':
                refined_codebook.append(code)
                print("✓ Code kept")
            elif action == 'm':
                # Allow modification
                print("\nEnter modifications (press Enter to keep current value):")
                
                new_category = input(f"Category [{code['category']}]: ").strip()
                if new_category:
                    code['category'] = new_category
                    
                new_description = input(f"Description [{code['description']}]: ").strip()
                if new_description:
                    code['description'] = new_description
                    
                new_codes = input(f"Keywords [{', '.join(code['codes'])}]: ").strip()
                if new_codes:
                    code['codes'] = [c.strip() for c in new_codes.split(',')]
                    
                new_inclusive = input(f"Include [{code['inclusive']}]: ").strip()
                if new_inclusive:
                    code['inclusive'] = new_inclusive
                    
                new_exclusive = input(f"Exclude [{code['exclusive']}]: ").strip()
                if new_exclusive:
                    code['exclusive'] = new_exclusive
                    
                refined_codebook.append(code)
                print("✓ Code modified and kept")
            elif action == 'd':
                print("✗ Code deleted")
            else:
                print("→ Code skipped for later review")
                
        # Option to add custom codes
        while True:
            add_custom = input("\nAdd a custom code? (y/n): ").lower()
            if add_custom == 'y':
                custom_code = {
                    'category': input("Category: "),
                    'description': input("Description: "),
                    'codes': [c.strip() for c in input("Keywords (comma-separated): ").split(',')],
                    'inclusive': input("Inclusive criteria: "),
                    'exclusive': input("Exclusive criteria: ")
                }
                refined_codebook.append(custom_code)
                print("✓ Custom code added")
            else:
                break
                
        return refined_codebook
    
    def calculate_code_scores(self, codebook: List[Dict], transcripts: Dict[str, List[srt.Subtitle]]) -> Dict:
        """
        Calculate similarity scores for each code against all transcripts
        
        Returns:
            Dictionary with code scoring metrics
        """
        code_scores = {}
        
        # Get embeddings for each code
        code_embeddings = []
        for code in codebook:
            # Combine code elements for embedding
            code_text = f"{code['category']} {code['description']} {' '.join(code['codes'])} {code['inclusive']}"
            inputs = self.bert_tokenizer(code_text, return_tensors="pt", 
                                       truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                code_embeddings.append(embedding[0])
        
        # Calculate scores for each transcript
        for session_id, subtitles in tqdm(transcripts.items(), desc="Scoring codes"):
            # Get transcript text
            transcript_text = " ".join([sub.content for sub in subtitles])
            
            # Get embedding
            inputs = self.bert_tokenizer(transcript_text[:512], return_tensors="pt", 
                                       truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                transcript_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Calculate similarities
            for i, code in enumerate(codebook):
                similarity = cosine_similarity([code_embeddings[i]], transcript_embedding)[0][0]
                
                if code['category'] not in code_scores:
                    code_scores[code['category']] = {
                        'similarities': [],
                        'mean_similarity': 0,
                        'std_similarity': 0,
                        'coverage': 0
                    }
                    
                code_scores[code['category']]['similarities'].append(similarity)
        
        # Calculate aggregate statistics
        for category, scores in code_scores.items():
            scores['mean_similarity'] = np.mean(scores['similarities'])
            scores['std_similarity'] = np.std(scores['similarities'])
            scores['coverage'] = len([s for s in scores['similarities'] if s > 0.7]) / len(scores['similarities'])
            
        return code_scores
    
    def generate_inductive_codebook(self, 
                                   transcripts: Dict[str, List[srt.Subtitle]], 
                                   n_codes: int = 20,
                                   interactive_refinement: bool = True) -> List[Dict]:
        """
        Main method to generate an inductive codebook from transcripts
        
        Args:
            transcripts: Dictionary of session transcripts
            n_codes: Target number of codes
            interactive_refinement: Whether to include researcher refinement
            
        Returns:
            Generated codebook
        """
        logger.info("Starting inductive codebook generation...")
        
        # Step 1: Generate summaries for each session
        all_participant_concepts = []
        all_instructor_concepts = []
        
        for session_id, subtitles in tqdm(transcripts.items(), desc="Processing sessions"):
            instructor_texts, participant_texts = self.separate_speakers(subtitles)
            
            if instructor_texts:
                instructor_summary = self.generate_session_summary(instructor_texts, "instructor")
                self.instructor_summaries[session_id] = instructor_summary
                instructor_concepts = self.extract_key_concepts(instructor_summary)
                all_instructor_concepts.extend(instructor_concepts)
            
            if participant_texts:
                participant_summary = self.generate_session_summary(participant_texts, "participant")
                self.participant_summaries[session_id] = participant_summary
                participant_concepts = self.extract_key_concepts(participant_summary)
                all_participant_concepts.extend(participant_concepts)
        
        # Step 2: Cluster concepts
        logger.info("Clustering concepts...")
        participant_clusters = self.cluster_concepts(all_participant_concepts, n_clusters=n_codes)
        
        # Step 3: Generate code definitions
        logger.info("Generating code definitions...")
        codebook = []
        
        for cluster_id, concepts in participant_clusters.items():
            # Try to match to theoretical domain
            domain = self._match_theoretical_domain(concepts)
            
            # Generate code definition
            code_def = self.generate_code_definition(concepts, domain)
            codebook.append(code_def)
        
        # Step 4: Interactive refinement
        if interactive_refinement:
            codebook = self.refine_codebook_with_researcher(codebook)
        
        # Step 5: Calculate code scores
        logger.info("Calculating code scores...")
        code_scores = self.calculate_code_scores(codebook, transcripts)
        
        # Add scores to codebook
        for code in codebook:
            if code['category'] in code_scores:
                code['score_metrics'] = code_scores[code['category']]
        
        # Save codebook
        self.codebook = codebook
        self.codebook_history.append({
            'timestamp': datetime.now().isoformat(),
            'codebook': codebook,
            'scores': code_scores
        })
        
        return codebook
    
    def _match_theoretical_domain(self, concepts: List[str]) -> Optional[str]:
        """Match concepts to MORE theoretical domains"""
        best_domain = None
        best_score = 0
        
        concept_text = " ".join(concepts).lower()
        
        for domain, keywords in self.theoretical_domains.items():
            score = sum(1 for keyword in keywords if keyword in concept_text)
            if score > best_score:
                best_score = score
                best_domain = domain
                
        return best_domain if best_score > 0 else None
    
    def apply_codebook_to_transcripts(self, 
                                     transcripts: Dict[str, List[srt.Subtitle]], 
                                     output_dir: str = "coded_output"):
        """
        Apply generated codebook to transcripts using apply_codebook.py
        
        Args:
            transcripts: Dictionary of session transcripts
            output_dir: Directory for coded output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert transcripts to SRT files for apply_codebook
        temp_dir = "temp_srt"
        os.makedirs(temp_dir, exist_ok=True)
        
        for session_id, subtitles in transcripts.items():
            srt_path = os.path.join(temp_dir, f"{session_id}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subtitles))
        
        # Apply codebook using the existing apply_codebook.py function
        all_coded_results = []
        
        for session_id in tqdm(transcripts.keys(), desc="Applying codebook"):
            srt_path = os.path.join(temp_dir, f"{session_id}.srt")
            
            coded_srt, stats, codes_applied = apply_research_codes_to_sentences(
                srt_file=srt_path,
                codes=self.codebook,
                coded_output_only=True,
                max_codes_per_sentence=3
            )
            
            # Save coded output
            output_path = os.path.join(output_dir, f"{session_id}_coded.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(coded_srt)
            
            all_coded_results.append({
                'session_id': session_id,
                'codes_applied': codes_applied,
                'stats': stats
            })
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        return all_coded_results
    
    def iterative_refinement(self, 
                           transcripts: Dict[str, List[srt.Subtitle]], 
                           n_iterations: int = 3) -> List[Dict]:
        """
        Perform iterative inductive-deductive-inductive refinement
        
        Args:
            transcripts: Dictionary of session transcripts
            n_iterations: Number of refinement iterations
            
        Returns:
            Final refined codebook
        """
        logger.info(f"Starting iterative refinement process ({n_iterations} iterations)...")
        
        for iteration in range(n_iterations):
            logger.info(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Generate or refine codebook
            if iteration == 0:
                # Initial inductive phase
                self.generate_inductive_codebook(
                    transcripts, 
                    n_codes=20,
                    interactive_refinement=True
                )
            else:
                # Apply current codebook (deductive phase)
                coded_results = self.apply_codebook_to_transcripts(transcripts)
                
                # Analyze coding results to identify gaps
                uncoded_segments = self._identify_uncoded_segments(coded_results, transcripts)
                
                # Generate new codes for uncoded segments (inductive phase)
                if uncoded_segments:
                    new_concepts = []
                    for segment in uncoded_segments[:50]:  # Limit to top 50
                        concepts = self.extract_key_concepts(segment)
                        new_concepts.extend(concepts)
                    
                    # Cluster and generate new codes
                    new_clusters = self.cluster_concepts(new_concepts, n_clusters=5)
                    
                    for cluster_id, concepts in new_clusters.items():
                        code_def = self.generate_code_definition(concepts)
                        self.codebook.append(code_def)
                
                # Researcher refinement
                self.codebook = self.refine_codebook_with_researcher(self.codebook)
            
            # Save iteration results
            self.save_codebook(f"codebook_iteration_{iteration + 1}.json")
        
        return self.codebook
    
    def _identify_uncoded_segments(self, 
                                  coded_results: List[Dict], 
                                  transcripts: Dict[str, List[srt.Subtitle]]) -> List[str]:
        """Identify transcript segments that weren't coded"""
        uncoded_segments = []
        
        for result in coded_results:
            session_id = result['session_id']
            codes_applied = result['codes_applied']
            
            # Get original transcript
            subtitles = transcripts[session_id]
            full_text = " ".join([sub.content for sub in subtitles])
            
            # Simple heuristic: segments with few codes
            if len(codes_applied) < len(subtitles) * 0.1:  # Less than 10% coded
                # Extract potentially important uncoded segments
                for sub in subtitles:
                    if len(sub.content.split()) > 20:  # Longer utterances
                        uncoded_segments.append(sub.content)
        
        return uncoded_segments
    
    def save_codebook(self, filename: str = "generated_codebook.json"):
        """Save codebook to JSON file"""
        output = {
            'generated_date': datetime.now().isoformat(),
            'n_codes': len(self.codebook),
            'codes': self.codebook,
            'metadata': {
                'n_sessions_analyzed': len(self.participant_summaries),
                'theoretical_domains': list(self.theoretical_domains.keys())
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Codebook saved to {filename}")
    
    def load_codebook(self, filename: str):
        """Load codebook from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.codebook = data['codes']
        
        logger.info(f"Loaded codebook with {len(self.codebook)} codes")


def main():
    """
    Main execution function demonstrating the complete workflow
    """
    # Initialize the generator
    generator = LlamaCodebookGenerator(
        model_path="meta-llama/Llama-3-70b-chat-hf",
        clinical_bert=True,
        quantization=True
    )
    
    # Load transcripts
    transcript_dir = "more_transcripts"
    transcripts = generator.load_transcripts(transcript_dir)
    
    # Option 1: Generate initial inductive codebook
    print("\n=== INDUCTIVE CODEBOOK GENERATION ===")
    codebook = generator.generate_inductive_codebook(
        transcripts,
        n_codes=20,
        interactive_refinement=True
    )
    
    # Save initial codebook
    generator.save_codebook("initial_inductive_codebook.json")
    
    # Option 2: Apply codebook and get results
    print("\n=== APPLYING CODEBOOK ===")
    coded_results = generator.apply_codebook_to_transcripts(transcripts)
    
    # Option 3: Perform iterative refinement
    print("\n=== ITERATIVE REFINEMENT ===")
    refined_codebook = generator.iterative_refinement(transcripts, n_iterations=3)
    
    # Save final codebook
    generator.save_codebook("final_refined_codebook.json")
    
    # Generate summary report
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report = f"""
    Mindfulness-Oriented Recovery Enhancement (MORE) Codebook Generation Report
    ==========================================================================
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Sessions Analyzed: {len(transcripts)}
    Final Codes: {len(refined_codebook)}
    
    Top Codes by Coverage:
    """
    
    # Sort codes by coverage
    sorted_codes = sorted(
        refined_codebook, 
        key=lambda x: x.get('score_metrics', {}).get('coverage', 0), 
        reverse=True
    )
    
    for i, code in enumerate(sorted_codes[:10]):
        metrics = code.get('score_metrics', {})
        report += f"""
    {i+1}. {code['category']}
       - Description: {code['description']}
       - Coverage: {metrics.get('coverage', 0):.2%}
       - Mean Similarity: {metrics.get('mean_similarity', 0):.3f}
    """
    
    # Save report
    with open("codebook_generation_report.txt", 'w') as f:
        f.write(report)
    
    print(report)
    print("\nCodebook generation complete!")


if __name__ == "__main__":
    main()
