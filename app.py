import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf

# ---------------------
# Load Model
# ---------------------
MODEL_PATH = "lstm_mfcc_model.h5"
model = load_model(MODEL_PATH)

st.title(" Deepfake Audio Detection App")
st.write("Upload a `.wav` file to detect whether it is **Real** or **Fake**.")

# ---------------------
# Helper Function
# ---------------------
def extract_mfcc_for_predict(path, sr=16000, duration=2.0, n_mfcc=40, max_len=128):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    expected_len = int(sr * duration)

    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc.T[np.newaxis, ...].astype('float32')  # shape (1, 128, 40)

# ---------------------
# Streamlit UI
# ---------------------
audio_file = st.file_uploader("Upload an audio file...", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Save uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Preprocess & Predict
    data = extract_mfcc_for_predict("uploaded_audio.wav")
    prediction = model.predict(data)[0][0]
    label = " REAL" if prediction < 0.5 else " FAKE"

    st.write("##  Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence Score:** {prediction:.4f}")
