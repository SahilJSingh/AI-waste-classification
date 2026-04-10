import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIGURATION (Tailored to PROJECT VER 2.ipynb) ---
SUBCATEGORIES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

HIERARCHY_MAP = {
    'metal': 'Recycling', 'glass': 'Recycling', 'plastic': 'Recycling',
    'paper': 'Recycling', 'cardboard': 'Recycling', 'biological': 'Organics',
    'clothes': 'Landfill', 'shoes': 'Landfill', 'trash': 'Landfill',
    'battery': 'Hazardous Waste',
}

# --- OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    # compile=False prevents the model from loading extra training parameters,
    # significantly reducing RAM usage to prevent Segmentation Faults.
    return tf.keras.models.load_model('waste_hierarchical_model.h5', compile=False)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- LIVE VIDEO PROCESSING ---
class WasteClassifier(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing: Resize to 224x224 and Rescale (1./255)
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        tensor = np.expand_dims(rgb, axis=0)

        # Prediction
        preds = model.predict(tensor, verbose=0)
        idx = np.argmax(preds)
        
        # Hierarchical Mapping
        subcat = SUBCATEGORIES[idx]
        parent = HIERARCHY_MAP.get(subcat, 'Unknown')
        confidence = np.max(preds) * 100

        # Draw results on video
        label = f"{subcat.capitalize()} ({parent}): {confidence:.1f}%"
        cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img

st.title("🎥 Live Hierarchical Waste Classifier")
st.write("Point your camera at waste to see the category and bin.")

webrtc_streamer(
    key="waste-scanner",
    video_transformer_factory=WasteClassifier,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
