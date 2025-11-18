import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("lstm_mfcc_model.h5")

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
    return mfcc.T[np.newaxis, ...].astype('float32')

st.title("🎤 Deepfake Audio Detection")
st.write("Upload a **.wav** file to check if it's REAL or FAKE.")

audio_file = st.file_uploader("Upload audio", type=["wav"])

if audio_file:
    st.audio(audio_file)

    # Save uploaded file
    with open("uploaded.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Predict
    x = extract_mfcc_for_predict("uploaded.wav")
    pred = model.predict(x)[0][0]
    label = "🟢 REAL" if pred < 0.5 else "🔴 FAKE"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {pred:.4f}")
