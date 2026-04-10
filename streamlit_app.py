import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIGURATION (Tailored to PROJECT VER 2.ipynb) ---
# Subcategories must match the alphabetical order of your dataset folders 
SUBCATEGORIES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Hierarchical mapping from your notebook 
HIERARCHY_MAP = {
    'metal': 'Recycling',
    'glass': 'Recycling',
    'plastic': 'Recycling',
    'paper': 'Recycling',
    'cardboard': 'Recycling',
    'biological': 'Organics',
    'clothes': 'Landfill',
    'shoes': 'Landfill',
    'trash': 'Landfill',
    'battery': 'Hazardous Waste',
}

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    # Load the .h5 file with compile=False for faster loading and fewer errors on Streamlit 
    model_path = 'waste_hierarchical_model.h5'
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- LIVE VIDEO PROCESSOR ---
class WasteClassifier(VideoTransformerBase):
    def transform(self, frame):
        # Convert the live frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing: Resize to 224x224 and Normalize 
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        tensor = np.expand_dims(rgb, axis=0)

        # Prediction logic
        preds = model.predict(tensor, verbose=0)
        idx = np.argmax(preds)
        
        # Map back to names
        subcat = SUBCATEGORIES[idx]
        parent = HIERARCHY_MAP.get(subcat, 'Unknown')
        confidence = np.max(preds) * 100

        # Draw results directly on the Live Video Stream
        text = f"{subcat.capitalize()} -> {parent} ({confidence:.1f}%)"
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img

# --- UI LAYOUT ---
st.set_page_config(page_title="Live Waste AI", layout="centered")
st.title("♻️ Real-Time Waste Classifier")
st.write("This application uses your live webcam to identify waste items and suggest the correct disposal bin.")

# Sidebar for manual testing
st.sidebar.title("Manual Test")
uploaded_file = st.sidebar.file_uploader("Or upload an image instead", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Handle manual upload if used
    st.subheader("Manual Upload Result")
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, width=300)
    
    # Preprocess uploaded image
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_tensor, verbose=0)
    idx = np.argmax(preds)
    sub = SUBCATEGORIES[idx]
    st.info(f"Detected: **{sub.capitalize()}** | Dispose in: **{HIERARCHY_MAP.get(sub, 'Unknown')}**")

# Start the Real-Time Stream
st.divider()
st.subheader("Live Scanner")
webrtc_streamer(
    key="waste-scanner", 
    video_transformer_factory=WasteClassifier,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
