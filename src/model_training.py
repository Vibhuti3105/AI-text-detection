import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data(data_path="data/processed/train.csv"):
    """Load and preprocess training data"""
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None

    try:
        df = pd.read_csv(data_path)
        texts = df['text'].values
        labels = df['label'].values  # 0 = human, 1 = AI
        return texts, labels
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def create_bert_model():
    """Create BERT-based classification model"""
    # BERT preprocessing layer
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    preprocess_layer = hub.KerasLayer(preprocess_url, name='preprocessing')

    # BERT encoder
    encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
    encoder_layer = hub.KerasLayer(encoder_url, trainable=True, name='BERT_encoder')

    # Model architecture
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = preprocess_layer(input_layer)
    encoder_outputs = encoder_layer(preprocessing_layer)

    # Use pooled output for classification
    pooled_output = encoder_outputs['pooled_output']

    # Classification head
    x = tf.keras.layers.Dropout(0.1)(pooled_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(input_layer, output_layer)
    return model

def train_model(model, train_texts, train_labels, val_texts, val_labels, epochs=3):
    """Train the BERT model"""
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        train_texts,
        train_labels,
        validation_data=(val_texts, val_labels),
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stopping]
    )

    return history

def save_model(model, save_path="saved_models/bert_detector"):
    """Save the trained model"""
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Load data
    texts, labels = load_data()
    if texts is None:
        print("No data found. Please prepare the dataset first.")
        exit()

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create and train model
    model = create_bert_model()
    print("Model created. Starting training...")

    history = train_model(model, train_texts, train_labels, val_texts, val_labels)

    # Save model
    save_model(model)
    print("Training completed!")