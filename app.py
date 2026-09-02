import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


# -----------------------------
# Load trained CNN model
# -----------------------------
model = tf.keras.models.load_model("digit_model.keras")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("✍️ Handwritten Digit Recognition")

st.write(
    "Draw a handwritten digit from 0 to 9 "
    "and let the CNN model recognize it."
)


# -----------------------------
# Drawing Canvas
# -----------------------------
st.subheader("Draw your digit")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Digit"):

    if canvas_result.image_data is not None:

        # Convert canvas to PIL image
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Convert to grayscale
        img = img.convert("L")

        # Resize to MNIST size
        img = img.resize((28, 28))

        # Convert to NumPy array
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array.astype("float32") / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(
            1, 28, 28, 1
        )

        # Make prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        # Get predicted digit
        predicted_digit = np.argmax(prediction)

        # Get confidence
        confidence = np.max(prediction) * 100


        # -----------------------------
        # Display prediction
        # -----------------------------
        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # -----------------------------
        # Probability
        # -----------------------------
        st.subheader("Prediction Probability")

        probabilities = prediction[0]

        st.bar_chart(probabilities)