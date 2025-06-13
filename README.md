# requirements.txt
torch>=2.0.0
transformers>=4.35.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
networkx>=3.1
scipy>=1.10.0
nltk>=3.8.0
srt>=3.5.0
tqdm>=4.65.0
PyYAML>=6.0
openpyxl>=3.1.0
reportlab>=4.0.0

---

# README.md

# MORE Qualitative Analysis Pipeline

An automated qualitative analysis system for Mindfulness-Oriented Recovery Enhancement (MORE) clinical research, featuring Llama3-70B powered codebook generation and machine learning-assisted coding.

## Overview

This pipeline implements an iterative inductive-deductive-inductive approach for qualitative analysis of mindfulness-based intervention sessions, specifically designed for chronic pain research.

### Key Features

- **Automated Codebook Generation**: Uses Llama3-70B to generate inductive codebooks from session transcripts
- **Theory-Driven Coding**: Integrates MORE theoretical domains (attention regulation, pain reappraisal, metacognitive awareness)
- **CueResponse Analysis**: Identifies therapeutic patterns between instructor cues and participant responses
- **Clinical Validation Tools**: Inter-rater reliability assessment and progression analysis
- **Multiple Export Formats**: Excel, CSV, REDCap data dictionary, and comprehensive PDF reports
- **HIPAA Compliant**: Local processing with encryption and de-identification

## System Requirements

- **Hardware**: 
  - AMD Threadripper Pro 5965WX or equivalent (24+ cores recommended)
  - 512GB RAM (minimum 256GB for Llama3-70B with quantization)
  - NVIDIA GPU with 48GB+ VRAM (for optimal performance)
  
- **Software**:
  - Python 3.9+
  - CUDA 11.8+ (for GPU acceleration)
  - Linux/Ubuntu (recommended) or Windows with WSL2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/more-analysis-pipeline
cd more-analysis-pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# Llama3-70B will be downloaded automatically on first use
# Alternatively, pre-download:
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-70b-chat-hf')"
```

## Configuration

Edit `config.yaml` to customize:
- Model paths and parameters
- Analysis settings (number of codes, clustering parameters)
- Theoretical domains specific to your study
- File paths for input/output

## Usage

### 1. Inductive Codebook Generation

Generate a codebook from scratch based on participant data:

```bash
python main.py inductive --apply --report
```

This will:
- Load all transcripts from the configured directory
- Generate summaries of instructor and participant dialogue
- Create an inductive codebook through clustering and LLM analysis
- Optionally apply the codebook and generate reports

### 2. Deductive Coding

Apply an existing codebook to new transcripts:

```bash
python main.py deductive --codebook codebooks/my_codebook.json --report
```

### 3. Iterative Refinement

Perform multiple cycles of inductive-deductive-inductive refinement:

```bash
python main.py iterative
```

### 4. Validation

Compare AI coding against human coding:

```bash
python main.py validate --human-codes human_coded.json --ai-codes ai_coded.json
```

### 5. CueResponse Analysis

Analyze instructor-participant interaction patterns:

```bash
python main.py cue-response
```

## File Structure

```
more-analysis-pipeline/
├── main.py                    # Main execution script
├── generate_codebook.py       # Llama3-based codebook generation
├── apply_codebook.py          # BERT-based code application
├── clinical_analysis_tools.py # Visualization and reporting
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── more_transcripts/         # Input transcript directory
│   ├── session_1.srt
│   ├── session_2.srt
│   └── ...
├── codebooks/               # Generated codebooks
├── analysis_output/         # Coded transcripts
└── clinical_reports/        # Analysis reports
```

## Transcript Format

Transcripts should be in SRT format with speaker identification:

```srt
1
00:00:00,000 --> 00:00:10,500
Instructor: Let's begin with a mindful breathing exercise...

2
00:00:11,000 --> 00:00:25,000
Participant: I noticed my mind wandering to my back pain...
```

## Theoretical Framework

The system is built around MORE's core therapeutic mechanisms:

1. **Attention Regulation**: Training focused attention and metacognitive monitoring
2. **Reappraisal**: Changing interpretations of pain sensations
3. **Savoring**: Enhancing natural reward processing

Codes are automatically mapped to these domains during generation.

## Clinical Outputs

### Generated Reports Include:

1. **Code Frequency Analysis**: Most common themes across sessions
2. **Therapeutic Progression**: How constructs evolve over time
3. **Network Analysis**: Code co-occurrence patterns
4. **Clinical Insights**: Actionable recommendations for intervention refinement

### Export Formats:

- **Excel**: Comprehensive codebook with metadata
- **CSV**: Simple format for statistical software
- **REDCap**: Data dictionary for clinical trials
- **PDF**: Complete clinical analysis report

## Validation Metrics

The system calculates:
- Cohen's Kappa (inter-rater reliability)
- Percent agreement
- Category-specific agreement
- Confusion matrices

Recommended thresholds:
- Cohen's Kappa ≥ 0.70
- Percent Agreement ≥ 80%

## Best Practices

1. **Interactive Refinement**: Always review generated codes with clinical expertise
2. **Iterative Approach**: Use 2-3 refinement cycles for optimal results
3. **Validation**: Compare a subset against human coding
4. **Documentation**: Keep notes on refinement decisions

## Troubleshooting

### Memory Issues
If encountering OOM errors with Llama3-70B:
- Enable 4-bit quantization (default)
- Reduce batch size in config
- Use CPU offloading if needed

### Poor Code Quality
- Increase number of clusters
- Add more specific theoretical domain keywords
- Use interactive refinement to guide the model

### Low Inter-rater Agreement
- Review exclusion criteria
- Refine code definitions
- Consider additional training examples

## Citation

If using this pipeline in research, please cite:

```bibtex
@software{more_analysis_pipeline,
  title={Automated Qualitative Analysis Pipeline for Mindfulness-Oriented Recovery Enhancement},
  author={Wade Balsamo},
  year={2026},
  url={https://github.com/wisgood/Qualitative_Research_Algorithm}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: research-support@your-institution.edu

## Acknowledgments

- MORE intervention developed by Dr. Eric Garland
- Built with Llama3 by Meta AI
- ClinicalBERT by Emily Alsentzer et al.
