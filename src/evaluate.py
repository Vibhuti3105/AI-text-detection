import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_model():
    """Load the trained BERT model"""
    if os.path.exists("saved_models/bert_detector"):
        try:
            model = tf.keras.models.load_model(
                "saved_models/bert_detector",
                custom_objects={'KerasLayer': hub.KerasLayer}
            )
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    else:
        print("Model not found. Please train the model first.")
        return None

def evaluate_model(model, test_texts, test_labels):
    """Evaluate the model on test data"""
    if model is None:
        return None

    try:
        # Make predictions
        predictions = model.predict(test_texts)
        pred_labels = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()

        # Calculate metrics
        print("Classification Report:")
        print(classification_report(test_labels, pred_labels))

        print("\nConfusion Matrix:")
        print(confusion_matrix(test_labels, pred_labels))

        # Calculate additional metrics
        accuracy = np.mean(pred_labels == test_labels)
        print(".4f")

        return {
            'predictions': pred_labels,
            'probabilities': pred_probs,
            'accuracy': accuracy
        }

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    model = load_model()

    if model is not None:
        # Example test data (replace with actual test data)
        test_texts = [
            "This is a human-written essay about artificial intelligence.",
            "As an AI language model, I can generate coherent and contextually appropriate text based on the input I receive."
        ]
        test_labels = [0, 1]  # 0 = human, 1 = AI

        results = evaluate_model(model, test_texts, test_labels)

