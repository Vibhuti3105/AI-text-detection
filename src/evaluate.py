import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # REQUIRED for BERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = tf.keras.models.load_model("saved_models/bert_detector", custom_objects={'KerasLayer': hub.KerasLayer})

print(model.summary())

# Load dataset
df = pd.read_csv("data/raw/train_essays.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns}")

X = df["text"].astype(str)
y = df["generated"].astype(int)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# SAME split logic as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Predict ONLY on test set
probs = model.predict(X_test, batch_size=8)

print(f"Probs shape: {probs.shape}")

# Assuming the model outputs (batch, 128, 1), take the first token (CLS)
if len(probs.shape) == 3:
    probs = probs[:, 0, :]

# Convert probabilities → labels
preds = (probs.flatten() > 0.5).astype(int)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, preds, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

