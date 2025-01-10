import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_emotion_model(model_path):
    """Load the pre-trained emotion classification model."""
    return load_model(model_path)

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_emotion(model, image_array, labels):
    """Predict the emotion using the pre-trained model."""
    predictions = model.predict(image_array)
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label

# Emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load model
MODEL_PATH = 'Models/model.keras'
model = load_emotion_model(MODEL_PATH)

# Streamlit app layout
st.title("Emotion Classification App")
st.write("Upload an image to classify the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    image_array = preprocess_image(image)
    predicted_emotion = predict_emotion(model, image_array, EMOTION_LABELS)

    # Display the predicted emotion
    st.subheader(f"Predicted Emotion: **{predicted_emotion}**")
