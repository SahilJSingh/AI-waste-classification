import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Waste Classifier AI", layout="centered")

# --- DATA CONFIG (Matching your PROJECT VER 2.ipynb) ---
SUBCATEGORIES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
                 'metal', 'paper', 'plastic', 'shoes', 'trash']

HIERARCHY_MAP = {
    'metal': 'Recycling', 'glass': 'Recycling', 'plastic': 'Recycling',
    'paper': 'Recycling', 'cardboard': 'Recycling', 'biological': 'Organics',
    'clothes': 'Landfill', 'shoes': 'Landfill', 'trash': 'Landfill',
    'battery': 'Hazardous Waste',
}

# --- MODEL LOADING WITH ERROR FEEDBACK ---
@st.cache_resource
def load_model():
    model_path = 'waste_hierarchical_model.h5'
    if not os.path.exists(model_path):
        return None
    try:
        # compile=False speeds up loading and prevents internal errors
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return None

# --- UI START ---
st.title("♻️ Smart Waste Classifier")

model = load_model()

if model is None:
    st.error("⚠️ Model file not found or corrupted. Please check your GitHub repository.")
    st.info("Ensure 'waste_hierarchical_model.h5' is in the root folder.")
    st.stop()

# --- INPUT SECTION ---
option = st.selectbox("How would you like to scan?", ("Upload an Image", "Use Webcam"))

if option == "Upload an Image":
    file = st.file_uploader("Pick an image...", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Scan item")

if file:
    # PREPROCESSING
    img = Image.open(file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # PREDICTION
    with st.spinner('Analyzing...'):
        preds = model.predict(img_tensor, verbose=0)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        
        sub = SUBCATEGORIES[idx]
        parent = HIERARCHY_MAP.get(sub, 'Unknown')

    # RESULTS DISPLAY
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, use_container_width=True)
        
    with col2:
        st.metric("Detected Item", sub.capitalize())
        st.header(f"Dispose as: :blue[{parent}]")
        st.write(f"Confidence: {confidence:.1f}%")
        
        if confidence < 50:
            st.warning("Low confidence. Ensure the item is well-lit and centered.")
