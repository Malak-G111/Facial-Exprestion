import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('your_model.h5')
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Streamlit UI
st.title("Facial Expression Recognition")
st.write("Upload an image to predict the emotion")

# Upload image from user
image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if image is not None:
    # Display uploaded image
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model input
    img = img.convert('L')  # Convert image to grayscale
    img = np.array(img)
    img = cv2.resize(img, (48, 48))  # Resize image
    img = img / 255.0  # Normalize the data
    img = img.reshape(1, 48, 48, 1)  # Reshape data to fit the model input

    # Predict the emotion
    predictions = model.predict(img)
    emotion = emotion_labels[np.argmax(predictions)]

    # Display result
    st.write(f"Predicted Emotion: {emotion}")