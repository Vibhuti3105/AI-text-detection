import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="AI Text Detector")

st.title("AI Text Detection")

def create_bert_model():
    """Create BERT-based classification model (same architecture as training)"""
    # BERT preprocessing layer
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    preprocess_layer = hub.KerasLayer(preprocess_url, name='preprocessing')

    # BERT encoder
    encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
    encoder_layer = hub.KerasLayer(encoder_url, trainable=True, name='BERT_encoder')

    # Model architecture (must match training script)
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

# Try to load model from Hugging Face
try:
    st.info("Loading model from Hugging Face...")

    # Download weights from Hugging Face
    weights_path = hf_hub_download(
        repo_id="vibhuti31/ai-text-detection",
        filename="bert_detector_weights.h5"
    )

    # Create model architecture
    model = create_bert_model()

    # Load the weights
    model.load_weights(weights_path)

    st.success("✅ Model loaded successfully from Hugging Face!")

    text = st.text_area("Enter text here", height=200)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            pred = model.predict([text])[0][0]
            st.write(f"### AI Probability: {pred:.2f}")

            if pred > 0.5:
                st.error("🤖 Likely AI-generated")
            else:
                st.success("👤 Likely Human-written")

except Exception as e:
    st.error(f"❌ Error loading model from Hugging Face: {str(e)}")
    st.info("💡 Alternative: Run locally with `streamlit run app/streamlit_app.py`")
    st.info("🔧 To use local model: Ensure 'saved_models/bert_detector' exists")
