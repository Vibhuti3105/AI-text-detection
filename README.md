# AI Text Detection

A machine learning application that detects whether text was written by AI or humans using a BERT-based model.

## Features

- **Real-time Analysis**: Enter any text and get instant AI probability scores
- **BERT Model**: Uses a fine-tuned BERT model for accurate detection
- **Streamlit Interface**: Clean, user-friendly web interface

## How to Use

1. Enter your text in the text area
2. Click "Analyze"
3. View the AI probability score and classification

## Model Details

- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Training Data**: Mixed dataset of human-written and AI-generated essays
- **Output**: Probability score between 0-1 (higher = more likely AI-generated)

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow 2.15.0
- **Model**: BERT via TensorFlow Hub
- **Language**: Python 3.10

## Local Development

```bash
# Clone the repository
git clone https://github.com/Vibhuti3105/AI-text-detection.git
cd AI-text-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

## Deployment

This app is deployed on Hugging Face Spaces for easy access.