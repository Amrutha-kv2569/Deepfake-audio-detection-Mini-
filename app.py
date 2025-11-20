import streamlit as st
import numpy as np
import os
import librosa
from tensorflow.keras.models import load_model
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# --- Configuration (MUST match training configuration exactly) ---
MODEL_FILENAME = 'best_cnn_spectrogram(cnn+lstm).h5' 
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
HOP_LENGTH = 512
N_MFCC = 40
PREDICTION_THRESHOLD = 0.5
TEMP_AUDIO_PATH = "uploaded_audio_temp.wav" # Use a consistent temp path

# --- 1. Calculate Required Feature Shape ---

def get_feature_shape(sr: int = SAMPLE_RATE, duration: float = DURATION, 
                      hop_length: int = HOP_LENGTH, n_mels: int = N_MELS, 
                      n_mfcc: int = N_MFCC) -> Tuple[int, int]:
    """Calculates the fixed feature shape (height, width) required by the CNN."""
    target_length = int(sr * duration)
    n_fft = 2048
    # Calculate width based on the fixed audio length
    time_steps = int(np.floor(1 + (target_length - n_fft) / hop_length))
    # Height is the combination of Mel (128) and MFCC (40)
    height = n_mels + n_mfcc
    return height, time_steps

# Calculate the required shape once
FEATURE_HEIGHT, FEATURE_WIDTH = get_feature_shape() 

# =================================================================
#                     PREPROCESSING FUNCTIONS (Mirroring Training Script)
# =================================================================

def normalize_feature_shape(feature: np.ndarray) -> np.ndarray:
    """Pads or truncates a feature array to the fixed target shape (168, 59)."""
    current_height, current_width = feature.shape

    if current_height != FEATURE_HEIGHT:
        # This should ideally not happen if extraction is correct
        raise ValueError(f"Height mismatch: Expected {FEATURE_HEIGHT}, got {current_height}")

    # Width (Time) check
    if current_width < FEATURE_WIDTH:
        padding = FEATURE_WIDTH - current_width
        feature = np.pad(feature, ((0, 0), (0, padding)), mode='constant')
    elif current_width > FEATURE_WIDTH:
        feature = feature[:, :FEATURE_WIDTH]
    
    return feature

def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract Mel Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    return mfcc

def preprocess_and_extract_features(file_path: str) -> Optional[np.ndarray]:
    """Handles full preprocessing pipeline for a single file."""
    try:
        # Load audio (mono, fixed SR, fixed duration)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure fixed audio length
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract and combine features
        mel_spec = extract_mel_spectrogram(audio, sr)
        mfcc = extract_mfcc(audio, sr)
        combined = np.vstack([mel_spec, mfcc]) # Shape (168, W)
        
        # Normalize feature shape to (168, 59)
        features = normalize_feature_shape(combined)
        
        # Add batch and channel dimensions: (168, 59) -> (1, 168, 59, 1)
        features = features[np.newaxis, ..., np.newaxis].astype('float32')
        
        return features
    
    except Exception as e:
        st.error(f"Error during audio processing: {e}")
        return None

# =================================================================
#                             MAIN APP LOGIC
# =================================================================

# --- Load Model (Cache to avoid re-loading) ---
@st.cache_resource
def load_deepfake_model():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Model file not found: {MODEL_FILENAME}. Please ensure it is in the same directory.")
        return None
    try:
        return load_model(MODEL_FILENAME)
    except Exception as e:
        st.error(f"Failed to load model from disk: {e}")
        return None

model = load_deepfake_model()

# --- Streamlit UI ---
st.title("Deepfake Audio Detection App")
st.write("Upload a `.wav` file to detect whether the speaker is **Real** or **Fake** (AI-generated).")
st.markdown(f"*(Model trained on Combined Mel Spectrogram + MFCC features with expected input shape: **(168, {FEATURE_WIDTH}, 1)**)*")

# File uploader
audio_file = st.file_uploader("Upload an audio file...", type=["wav"])

if model is None:
    st.stop()

if audio_file is not None:
    # 1. Display Audio
    st.audio(audio_file, format="audio/wav")
    
    # 2. Save uploaded file temporarily
    with open(TEMP_AUDIO_PATH, "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.write("---")
    st.spinner("Processing and predicting...")
    
    # 3. Preprocess & Predict
    input_data = preprocess_and_extract_features(TEMP_AUDIO_PATH)
    
    if input_data is not None:
        try:
            # Prediction returns a single probability (Fake confidence)
            prediction = model.predict(input_data, verbose=0)[0][0]
            
            label = "FAKE (AI-Generated)" if prediction >= PREDICTION_THRESHOLD else "REAL (Genuine)"
            
            # 4. Display Results
            st.write("##  Prediction Result")
            
            if label.startswith("FAKE"):
                st.error(f"The audio is predicted to be **{label}**")
            else:
                st.success(f"The audio is predicted to be **{label}**")
                
            st.markdown(f"**Confidence Score (Fake):** `{prediction:.4f}`")
            st.markdown(f"*(Prediction Threshold: {PREDICTION_THRESHOLD})*")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # 5. Clean up temporary file
    if os.path.exists(TEMP_AUDIO_PATH):
        os.remove(TEMP_AUDIO_PATH)
