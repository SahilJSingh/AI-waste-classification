import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. HIERARCHY CONFIGURATION (From PROJECT VER 2.ipynb) ---
# Must match your folder names exactly
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

# --- 2. OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    # compile=False reduces RAM usage significantly to prevent Segmentation Faults
    model = tf.keras.models.load_model('waste_hierarchical_model.h5', compile=False)
    return model

# Initialize model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

# --- 3. LIVE VIDEO PROCESSING CLASS ---
class WasteTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to array
        img = frame.to_ndarray(format="bgr24")
        
        # PREPROCESSING: Resize and Rescale (1./255)
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        tensor = np.expand_dims(rgb, axis=0)

        # PREDICTION
        preds = model.predict(tensor, verbose=0)
        idx = np.argmax(preds)
        
        # HIERARCHICAL LOOKUP
        subcat = SUBCATEGORIES[idx]
        parent = HIERARCHY_MAP.get(subcat, 'Unknown')
        conf = np.max(preds) * 100

        # OVERLAY RESULTS ON LIVE FEED
        label = f"{subcat.capitalize()} -> {parent} ({conf:.1f}%)"
        cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI Waste Scanner", layout="centered")
st.title("🎥 Live Hierarchical Waste Classifier")
st.write("Point your camera at an object to see its subcategory and disposal bin.")

# The ICE Servers help establish the video connection through firewalls
webrtc_streamer(
    key="waste-live",
    video_transformer_factory=WasteTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.markdown("### Model Details")
st.sidebar.write("Backbone: MobileNetV2")
st.sidebar.write("Input Size: 224x224")
