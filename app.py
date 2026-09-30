import io
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.io import wavfile
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

st.set_page_config(page_title="Nigerian Accent Speech-to-Text", page_icon="🎤", layout="centered")

MODEL_ID = "openai/whisper-small"
LOCAL_CHECKPOINT = "finetuned_on_odia_steps_1200.pth"
CHECKPOINT_REPO = os.getenv("CHECKPOINT_REPO", "")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "model.pth")

LANGUAGE_OPTIONS = {
    "English": "english",
    "Auto-detect": None,
    "Yoruba": "yoruba",
    "Hausa": "hausa",
    "Igbo": "igbo",
}


def resolve_checkpoint():
    local = Path(__file__).with_name(LOCAL_CHECKPOINT)
    if local.exists():
        return str(local)
    if not CHECKPOINT_REPO:
        raise FileNotFoundError(
            f"No local '{LOCAL_CHECKPOINT}' found and CHECKPOINT_REPO is not set. "
            "Upload the checkpoint to the Hugging Face Hub and set the "
            "CHECKPOINT_REPO environment variable (e.g. 'your-user/whisper-nigerian')."
        )
    return hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE)


@st.cache_resource
def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.load_state_dict(torch.load(resolve_checkpoint(), map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model, device


def prepare_audio(audio_data):
    if hasattr(audio_data, "getvalue"):
        raw = audio_data.getvalue()
    elif hasattr(audio_data, "read"):
        audio_data.seek(0)
        raw = audio_data.read()
    else:
        raw = audio_data
    try:
        sample_rate, audio = wavfile.read(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not decode the recording: {exc}")
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)
    return resample_audio(audio, int(sample_rate), 16000)


def transcribe(audio_chunk, processor, model, device, language="english"):
    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to(device), **gen_kwargs)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return resample(audio, num_samples).astype(np.float32)


st.title("🎤 Nigerian Accent Speech-to-Text")
st.caption("Fine-tuned Whisper model for Nigerian accented speech recognition")

try:
    with st.spinner("Loading model..."):
        processor, model, device = load_model()
    st.success(f"Model loaded on **{device.upper()}**")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

st.markdown("---")

language_label = st.selectbox("Language", list(LANGUAGE_OPTIONS))
language = LANGUAGE_OPTIONS[language_label]

if "history" not in st.session_state:
    st.session_state.history = []

audio_value = st.audio_input("🎙️ Record audio")

col1, col2 = st.columns(2)

transcribe_clicked = col1.button("🔤 Transcribe", use_container_width=True)

with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []

if transcribe_clicked:
    if audio_value is None:
        st.warning("Record some audio first.")
    else:
        try:
            audio_16k = prepare_audio(audio_value)

            with st.spinner("Transcribing..."):
                start = time.time()
                text = transcribe(audio_16k, processor, model, device, language)
                elapsed = time.time() - start

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.insert(0, {"time": ts, "text": text, "duration": f"{elapsed:.2f}s"})

            st.subheader("Transcription")
            st.write(text)
            st.caption(f"Transcribed in {elapsed:.2f}s")

        except Exception as e:
            st.error(f"Transcription failed: {e}")

if st.session_state.history:
    st.markdown("---")
    st.subheader("History")
    for entry in st.session_state.history:
        with st.expander(f"[{entry['time']}] — {entry['duration']}"):
            st.write(entry["text"])
