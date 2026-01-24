import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import re

def load_raw_data(data_dir="data/raw"):
    """Load raw datasets from multiple sources"""
    all_texts = []
    all_labels = []

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return None, None

    # Load human-written essays
    human_files = ["train_essays.csv", "train_essays_RDizzl3_seven_v1.csv"]

    for file in human_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Assuming human essays have label 0
                texts = df['text'].fillna('').values
                labels = np.zeros(len(texts))
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"Loaded {len(texts)} human essays from {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    # Load AI-generated essays
    ai_files = ["LLM_generated_essay_PaLM.csv", "falcon_180b_v1.csv"]

    for file in ai_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Assuming AI essays have label 1
                texts = df['text'].fillna('').values
                labels = np.ones(len(texts))
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"Loaded {len(texts)} AI essays from {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    return all_texts, all_labels

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text

def preprocess_data(texts, labels, max_length=512):
    """Preprocess the dataset"""
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]

    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 10]
    cleaned_texts = [cleaned_texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    # Truncate long texts (BERT has 512 token limit)
    processed_texts = []
    for text in cleaned_texts:
        if len(text) > max_length * 4:  # Rough character to token ratio
            text = text[:max_length * 4] + "..."
        processed_texts.append(text)

    return processed_texts, labels

def create_balanced_dataset(texts, labels, max_samples_per_class=None):
    """Create a balanced dataset"""
    human_indices = [i for i, label in enumerate(labels) if label == 0]
    ai_indices = [i for i, label in enumerate(labels) if label == 1]

    print(f"Human samples: {len(human_indices)}")
    print(f"AI samples: {len(ai_indices)}")

    # Balance the dataset
    min_samples = min(len(human_indices), len(ai_indices))

    if max_samples_per_class:
        min_samples = min(min_samples, max_samples_per_class)

    # Randomly sample
    np.random.seed(42)
    human_sample = np.random.choice(human_indices, min_samples, replace=False)
    ai_sample = np.random.choice(ai_indices, min_samples, replace=False)

    balanced_indices = np.concatenate([human_sample, ai_sample])
    np.random.shuffle(balanced_indices)

    balanced_texts = [texts[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]

    return balanced_texts, balanced_labels

def save_processed_data(texts, labels, output_path="data/processed/train.csv"):
    """Save processed data to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })

    df.to_csv(output_path, index=False)
    print(f"Saved {len(texts)} samples to {output_path}")

if __name__ == "__main__":
    print("Loading raw data...")
    texts, labels = load_raw_data()

    if texts is None:
        print("No data found. Please check the data/raw directory.")
        exit()

    print(f"Total samples loaded: {len(texts)}")

    print("Preprocessing data...")
    processed_texts, processed_labels = preprocess_data(texts, labels)

    print("Creating balanced dataset...")
    balanced_texts, balanced_labels = create_balanced_dataset(processed_texts, processed_labels)

    print(f"Balanced dataset: {len(balanced_texts)} samples")

    print("Saving processed data...")
    save_processed_data(balanced_texts, balanced_labels)

    print("Dataset preparation completed!")