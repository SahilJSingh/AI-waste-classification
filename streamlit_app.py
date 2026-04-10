import numpy as np
from PIL import Image
import tensorflow as tf

# 1. Matching the Configuration from Cell 2 of your Notebook
IMG_SIZE = (224, 224)
# Ensure this matches the exact alphabetical order of your subfolders
SUBCATEGORIES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
                 'metal', 'paper', 'plastic', 'shoes', 'trash']

# 2. Matching the Hierarchy Map from Cell 2
HIERARCHY_MAP = {
    'metal': 'Recycling', 'glass': 'Recycling', 'plastic': 'Recycling',
    'paper': 'Recycling', 'cardboard': 'Recycling', 'biological': 'Organics',
    'clothes': 'Landfill', 'shoes': 'Landfill', 'trash': 'Landfill',
    'battery': 'Hazardous Waste',
}

def preprocess_and_predict(uploaded_file, model):
    # Step A: Load image using PIL (Streamlit standard)
    img = Image.open(uploaded_file)
    
    # Step B: Resize to (224, 224) - Matching IMG_SIZE in Notebook
    img = img.resize(IMG_SIZE)
    
    # Step C: Convert to Array and NORMALIZE (Critical: matching rescale=1./255)
    img_array = np.array(img).astype('float32') / 255.0
    
    # Step D: Add Batch Dimension (Model expects [1, 224, 224, 3])
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Step E: Prediction
    preds = model.predict(img_tensor, verbose=0)
    pred_idx = np.argmax(preds)
    
    # Step F: Hierarchical Logic (Matching Cell 6)
    subcategory = SUBCATEGORIES[pred_idx]
    parent_category = HIERARCHY_MAP.get(subcategory, 'Unknown')
    confidence = np.max(preds) * 100
    
    return subcategory, parent_category, confidence
