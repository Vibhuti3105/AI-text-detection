# AI-Generated Text Detection using BERT

A machine learning system that detects whether a given text is human-written or AI-generated using a fine-tuned BERT-based sequence classification model. The project focuses on black-box AI-text detection, robustness analysis, and real-world failure modes.

## ✨ Key Features

**Black-Box AI-Text Detection**
Detects AI-generated text without relying on watermarking or model internals.

**BERT-Based Classification**
Fine-tuned BERT model captures semantic, lexical, and syntactic patterns to distinguish human and LLM-generated text.

**Robust Dataset Curation**
Trained on 35,500+ samples aggregated from multiple public datasets, including adversarial and diverse writing styles.

**Interactive Web Interface**
Streamlit-based UI for real-time inference and probability scoring.

## 🧠 How It Works (High Level)

**Preprocessing**
Text is normalized and tokenized using BERT-compatible preprocessing.

**Model Inference**
A fine-tuned BERT sequence classification model predicts the probability of AI-generated content.

**Decision Layer**
Outputs a probability score (0–1) along with a human/AI classification, optimized for low false-positive rates.

## 📊 Model Details

**Architecture**: BERT (Bidirectional Encoder Representations from Transformers)

**Training Data**: Mixed dataset of human-written and AI-generated essays

**Dataset Size**: 35,500+ samples

**Evaluation Metric**: F1-Score (98.2% on benchmark datasets)

**Output**: Probability score ∈ [0, 1]

## ⚠️ Known Limitations & Failure Modes

Performance degrades on heavily corrupted or typo-rich text, where token embeddings become unreliable.

Distribution shift between training and unseen test data can affect confidence calibration.

In some noisy scenarios, simpler lexical models (e.g., TF-IDF) may outperform deep models.

These observations highlight the importance of data quality and robustness analysis in AI-text detection systems.

## 🛠️ Tech Stack

**Frontend**: Streamlit

**Backend**: TensorFlow 2.15

**Model**: BERT via TensorFlow Hub

**Language**: Python 3.10

## 🚀 Local Development

```bash
# Clone the repository
git clone https://github.com/Vibhuti3105/AI-text-detection.git
cd AI-text-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

## 🌐 Deployment

The application is deployed on Streamlit Cloud for easy access and experimentation.