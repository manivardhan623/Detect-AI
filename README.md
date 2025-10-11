# AI Text Detection System

An advanced machine learning project that distinguishes AI-generated text from human-written content using hybrid linguistic analysis and deep learning.

## Overview

This project develops a robust AI text detection system by integrating diverse datasets and employing advanced feature engineering with the Human Language Tool Kit (HLTK) combined with transformer-based models.

## Features

- **Multi-Domain Dataset Integration**: Combines 6 distinct text sources covering social media, academic writing, creative content, and AI-generated text
- **Balanced Training Data**: Maintains 50-50 split between human and AI-generated samples
- **Advanced Linguistic Analysis**: Extracts perplexity, semantic coherence, syntactic complexity using HLTK
- **Hybrid Detection Model**: Ensemble approach combining Random Forest and ModernBERT transformers
- **Minimum Input Requirement**: Requires at least 150 words for reliable detection

## Dataset Sources

### Human Text (50%)
- **Casual/Social (20%)**: [Reddit Dataset](https://www.kaggle.com/datasets/pavellexyr/the-reddit-dataset-dataset)
- **Professional/Academic (20%)**: [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
- **Creative/Personal (10%)**: [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)

### AI-Generated Text (50%)
- **Modern LLMs (30%)**: [LLM Detection Dataset](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset)
- **Older Models (15%)**: [GPT Reddit Dataset](https://arxiv.org/html/2403.07321v1)
- **Specialized AI (5%)**: [DAIGT Dataset](https://github.com/iamjr15/Ensemble-AI-Text-Detection)

## Technical Stack

- **Language**: Python 3.11+
- **Libraries**: 
  - PyTorch / TensorFlow
  - Transformers (Hugging Face)
  - Scikit-learn
  - Pandas, NumPy
  - textstat (for linguistic metrics)
  - HLTK (Human Language Tool Kit)

## Model Architecture

Input Text (150+ words)
↓
Text Preprocessing & Cleaning
↓
Feature Extraction
├── Stylometric Features (15 metrics)
│ ├── Lexical diversity
│ ├── Avg word/sentence length
│ ├── Punctuation patterns
│ └── Readability scores
└── Perplexity Features (3 metrics)
├── Language model perplexity
├── Token entropy
└── Loss values
↓
ModernBERT Base (Transformer)
├── 12 Encoder Layers
├── 768 Hidden Dimensions
└── Multi-Head Self-Attention
↓
Feature Fusion Layer
├── BERT Embeddings (768-dim)
└── Linguistic Features (18-dim)
↓
Attention Mechanism (8 heads)
↓
Classification Head
├── Dense Layers (256 → 128 → 2)
└── Temperature Scaling
↓
Output: AI (60%+ threshold) or Human


## Installation

Clone the repository
git clone https://github.com/yourusername/ai-text-detector.git
cd ai-text-detector

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

## Usage

### Interactive Detection

### Python API

from detector import Detector

detector = Detector(
model_path='path/to/model',
bert_model='answerdotai/ModernBERT-base'
)

text = "Your text here (minimum 150 words)..."
result = detector.predict(text)

print(f"Prediction: {result['prediction']}")
print(f"AI Probability: {result['ai_prob']:.2%}")
print(f"Human Probability: {result['human_prob']:.2%}")

text

## Project Structure

ai-text-detector/
├── original-data/ # Raw datasets
├── ultimate_ai_detector/ # Trained model weights
│ ├── config.json
│ ├── pytorch_model.bin
│ └── tokenizer files
├── combined_train.csv # Preprocessed training data
├── combined_test.csv # Preprocessed test data
├── preprocess_and_split.py # Data preprocessing script
├── ultimate_ai_text_detector_train.py # Training script
├── test_ai_text_detector.py # Evaluation/inference script
├── requirements.txt
└── README.md

text

## Model Performance

- **Validation Accuracy**: ~100% (note: small dataset, potential overfitting)
- **Detection Threshold**: 60% AI probability for classification
- **Best Performance**: Texts with 150+ words
- **Limitations**: May misclassify heavily paraphrased AI text

## Training

python ultimate_ai_text_detector_train.py

text

### Training Configuration
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler**: Linear warmup + decay
- **Batch Size**: 16
- **Epochs**: 3
- **Loss**: Cross-Entropy with label smoothing (0.1)

## Key Components

### Feature Extraction
- **Stylometric Analysis**: Word/sentence length, lexical diversity, punctuation density
- **Readability Metrics**: Flesch Reading Ease, Flesch-Kincaid Grade, ARI
- **Syntactic Complexity**: Syllable counts, function word ratios
- **Perplexity Calculation**: Language model confidence scores

### Detection Strategy
- **Hybrid Approach**: Combines transformer embeddings with engineered features
- **Multi-head Attention**: 8 attention heads for feature fusion
- **Temperature Scaling**: Calibrates output probabilities
- **Threshold-based Classification**: Requires 60%+ AI probability

## Limitations & Future Work

### Current Limitations
- **Minimum Length Requirement**: Unreliable below 150 words
- **Paraphrase Vulnerability**: Struggles with humanized AI text
- **Small Training Set**: Risk of overfitting
- **Domain Bias**: May favor training distribution

### Planned Improvements
- Augment training with paraphrased AI examples
- Add adversarial training for robustness
- Expand dataset size and diversity
- Implement ensemble voting across multiple models
- Real-time detection API
- Support for multilingual text

## References

1. DevlinModernBERT: A Bidirectional Encoder for Language Understanding
2. BERT: Pre-training of Deep Bidirectional Transformers
3. [WikiText Language Model Dataset](https://huggingface.co/datasets/Salesforce/wikitext)
4. [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
5. [Kaggle LLM Detection Challenge](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

## Contributors

- [Your Name] - Project Lead & Development
- [Team Members] - Dataset Curation & Feature Engineering
- [Guide Name] - Project Advisor

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KL Hyderabad Off Campus, CSE Department
- Dataset providers: Kaggle, Hugging Face, UCSD, ArXiv
- ModernBERT model by Answer.AI

## Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
