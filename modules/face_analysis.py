# ======================================================
# 1. Import Libraries
# ======================================================
import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

# Import layers needed to rebuild the model architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# Updated import from MobileNetV2 to MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Small

# ======================================================
# 2. Define Constants
# ======================================================
# Use a relative path for the model, assuming it's in the same directory or a known location
MODEL_PATH = r"static/gender_model_finetuned.keras"  # Trained gender model weights
IMG_SIZE = (224, 224) # Define image size for the model input

# ======================================================
# 3. Load Gender Detection Model
# ======================================================
def build_model(weights_path=MODEL_PATH):
    """
    Loads the complete, finetuned Keras gender model from a single file.
    
    (Reverted to load_model() as the .keras file appears to be a full model,
    not just weights, which was causing the 'file signature' error.)
    """
    try:
        if not os.path.exists(weights_path):
            print(f"❌ Error: Model file not found at {weights_path}")
            print("Please make sure 'best_gender_model_e12_v13.keras' is in the static/ directory.")
            return None

        print(f"Loading complete gender detection model from {weights_path}...")
        
        # Use load_model() as the file is a complete model, not just weights
        model = tf.keras.models.load_model(weights_path)

        print("✅ Gender detection model loaded successfully.")
        print(f"📐 Model Input Shape: {model.input_shape}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading gender model: {e}")
        return None

# ======================================================
# 4. Gender Analysis Function
# ======================================================
def analyze_gender(face_crop, model, confidence_threshold=0.7):
    """
    Analyzes a single face crop and returns the predicted gender.
    
    Args:
        face_crop: The image (numpy array) of the face.
        model: The loaded TensorFlow/Keras gender model.
        confidence_threshold: Min confidence to return 'Male'/'Female'.
                              Returns 'Unknown' if confidence is too low.
                              
    Returns:
        A string: "Male", "Female", or "Unknown".
    """
    
    # Check if model is loaded
    if model is None:
        print("❌ Gender model is not loaded. Cannot analyze.")
        return "Unknown"
        
    # Check if face crop is valid
    if face_crop is None or face_crop.size == 0:
        print("⚠️ Warning: Empty face crop passed to analyze_gender.")
        return "Unknown"

    try:
        # Preprocess face for the model
        # Use the IMG_SIZE constant, as model.input_shape might be (None, 224, 224, 3)
        target_size = IMG_SIZE 
        face_resized = cv2.resize(face_crop, target_size)
        face_rescaled = face_resized / 255.0
        face_batch = np.expand_dims(face_rescaled, axis=0) # Create batch of 1

        # Predict gender
        # Assumes model output: 0.0 = Female, 1.0 = Male
        prediction = model.predict(face_batch, verbose=0)[0][0]

        # Determine label based on confidence
        if prediction > confidence_threshold:
            # Confidently Male (e.g., > 0.7)
            return "Male"
        elif prediction < (1 - confidence_threshold):
            # Confidently Female (e.g., < 0.3)
            return "Female"
        else:
            # Not confident (e.g., between 0.3 and 0.7)
            return "Unknown"

    except Exception as e:
        print(f"[analyze_gender] Prediction error: {e}")
        return "Unknown"

