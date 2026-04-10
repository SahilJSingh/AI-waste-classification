import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIGURATION ---
SUBCATEGORIES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

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

COLOR_MAP = {
    'Recycling': (0, 255, 0),
    'Organics': (0, 200, 100),
    'Landfill': (0, 165, 255),
    'Hazardous Waste': (0, 0, 255),
}

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path='waste_model.tflite')
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path='waste_model.tflite')

    interpreter.allocate_tensors()
    return interpreter


try:
    interpreter = load_trained_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except FileNotFoundError:
    st.error("❌ Model file 'waste_model.tflite' not found. Please make sure it's in the repo root.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# --- INFERENCE ---
def predict(img_tensor):
    interpreter.set_tensor(input_details[0]['index'], img_tensor.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


# --- VIDEO TRANSFORMER ---
class WasteClassifier(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        tensor = np.expand_dims(rgb, axis=0)

        # Predict
        preds = predict(tensor)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        # Map to category
        subcat = SUBCATEGORIES[idx]
        parent = HIERARCHY_MAP.get(subcat, 'Unknown')
        color = COLOR_MAP.get(parent, (255, 255, 255))

        # Overlay label
        label = f"{subcat.capitalize()} ({parent}): {confidence:.1f}%"
        cv2.rectangle(img, (20, 20), (len(label) * 11 + 30, 65), (0, 0, 0), -1)
        cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --- UI ---
st.title("♻️ Live Hierarchical Waste Classifier")
st.write("Point your camera at an item to classify it and find the right bin.")

# Bin legend
with st.expander("📋 Bin Guide", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.success("♻️ **Recycling**\nmetal, glass, plastic, paper, cardboard")
    col2.info("🌱 **Organics**\nbiological waste")
    col3.warning("🗑️ **Landfill**\nclothes, shoes, trash")
    col4.error("⚠️ **Hazardous**\nbatteries")

st.divider()

# WebRTC stream
webrtc_streamer(
    key="waste-scanner",
    video_transformer_factory=WasteClassifier,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.caption("Model: MobileNetV2-based hierarchical waste classifier | 10 categories")
