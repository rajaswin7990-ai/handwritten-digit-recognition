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

        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Convert to grayscale
        img = img.convert("L")

        # IMPORTANT: invert the image
        img = Image.eval(img, lambda x: 255 - x)

        # Resize to MNIST 28x28
        img = img.resize((28, 28))

        # Convert to NumPy
        img_array = np.array(img)

        # Normalize
        img_array = img_array.astype("float32") / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader("Prediction Probability")

        probabilities = prediction[0]

        st.bar_chart(probabilities)