import os
os.makedirs("saved_models", exist_ok=True)

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/train_essays.csv")
df = df[['text', 'generated']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['generated'], test_size=0.1, random_state=42
)

# BERT paths (TF Hub online)
preprocess = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
    trainable=True
)

# Model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

encoder_inputs = preprocess(text_input)

outputs = encoder(
    [
        encoder_inputs["input_word_ids"],
        encoder_inputs["input_mask"],
        encoder_inputs["input_type_ids"],
    ],
    training=True
)

net = outputs[1]  # pooled_output

net = tf.keras.layers.Dropout(0.2)(net)
net = tf.keras.layers.Dense(64, activation="relu")(net)
net = tf.keras.layers.Dense(1, activation="sigmoid")(net)

model = tf.keras.Model(text_input, net)



model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=1,
    batch_size=8
)

# Save model
model.save("saved_models/bert_detector")

print("Model saved successfully.")
