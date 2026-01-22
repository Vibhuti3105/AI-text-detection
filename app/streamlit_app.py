import streamlit as st
import tensorflow as tf
import tensorflow_text

st.set_page_config(page_title="AI Text Detector")

model = tf.keras.models.load_model("saved_models/bert_detector")

st.title("AI Text Detection")
st.write("Paste text below to check if it is AI-generated.")

text = st.text_area("Enter text here")

if st.button("Analyze"):
    pred = model.predict([text])[0][0].item()
    st.write(f"### AI Probability: {pred:.2f}")
    if pred > 0.5:
        st.error("Likely AI-generated")
    else:
        st.success("Likely Human-written")
