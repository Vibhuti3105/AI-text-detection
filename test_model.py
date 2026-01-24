import tensorflow as tf
import tensorflow_hub as hub
import os

if os.path.exists("saved_models/bert_detector"):
    try:
        model = tf.keras.models.load_model("saved_models/bert_detector", custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully!")
        # Test prediction
        pred = model.predict(["This is a test text."])
        print(f"Test prediction: {pred}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print("Model not found.")