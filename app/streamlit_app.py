import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import os

st.set_page_config(page_title="AI Text Detector")

st.title("AI Text Detection")

if os.path.exists("saved_models/bert_detector"):
    try:
        model = tf.keras.models.load_model("saved_models/bert_detector", custom_objects={'KerasLayer': hub.KerasLayer})
        st.write("Model loaded successfully!")
        
        text = st.text_area("Enter text here")
        
        if st.button("Analyze"):
            pred = model.predict([text])[0][0].item()
            st.write(f"### AI Probability: {pred:.2f}")
            if pred > 0.5:
                st.error("Likely AI-generated")
            else:
                st.success("Likely Human-written")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.error("Model not found. Please train the model using the scripts in the src/ folder or download a pre-trained model.")
