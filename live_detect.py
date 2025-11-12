import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import wavio

# Load your trained model
model = load_model("lstm_mfcc_model.h5")

# Function to record audio
def record_audio(duration=2, sr=16000):
    st.info("🎤 Recording... Speak now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    wavio.write("live_audio.wav", recording, sr, sampwidth=2)
    st.success("✅ Recording complete!")
    return "live_audio.wav"

# Extract MFCCs (same as training)
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

# Streamlit app
st.title("🎤 Live Deepfake Audio Detection")
st.write("Click the button below to record a 2-second audio sample and detect if it's real or fake.")

if st.button("🎙️ Record and Detect"):
    audio_path = record_audio()
    data = extract_mfcc_for_predict(audio_path)
    pred = model.predict(data)[0][0]
    label = "🟢 REAL" if pred < 0.5 else "🔴 FAKE"
    st.audio(audio_path)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence Score:** {pred:.4f}")
