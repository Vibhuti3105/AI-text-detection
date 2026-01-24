import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import os

st.set_page_config(page_title="AI Text Detector")

st.title("AI Text Detection")

# Load model from local files only
if os.path.exists("saved_models/bert_detector"):
    try:
        st.info("🔄 Loading model from local files...")
        model = tf.keras.models.load_model("saved_models/bert_detector", custom_objects={'KerasLayer': hub.KerasLayer})
        st.success("✅ Model loaded successfully!")

        text = st.text_area("Enter text here", height=200)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                pred = model.predict([text])[0][0].item()
                st.write(f"### AI Probability: {pred:.2f}")

                if pred > 0.5:
                    st.error("🤖 Likely AI-generated")
                else:
                    st.success("👤 Likely Human-written")

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Try running the training script: `python src/model_training.py`")
else:
    st.error("❌ Model not found!")
    st.info("🔧 Run: `python src/model_training.py` to train the model")
