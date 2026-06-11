import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array

# PAGE SETTINGS
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🧠",
    layout="centered"
)

# TITLE
st.title("🧠 AI Image Captioning App")

st.write(
    "Upload an image and the AI model will generate a caption."
)

# LOAD PRETRAINED VGG16 MODEL
model = VGG16(weights='imagenet')

# IMAGE PREPROCESSING FUNCTION
def preprocess_image(image):

    image = image.resize((224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image

# CAPTION GENERATION FUNCTION
def generate_caption(image):

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)

    decoded = decode_predictions(predictions, top=1)[0][0]

    caption = decoded[1].replace("_", " ").title()

    confidence = round(decoded[2] * 100, 2)

    return caption, confidence

# IMAGE UPLOADER
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# PROCESS IMAGE
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("Generating caption...")

    caption, confidence = generate_caption(image)

    st.success(f"Predicted Caption: {caption}")

    st.info(f"Confidence Score: {confidence}%")

else:

    st.warning("Please upload an image file.")