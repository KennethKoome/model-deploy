# the app
import streamlit as st
import pickle
import re
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Config
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "lstm_sentiment.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pickle"
LE_PATH = MODEL_DIR / "label_encoder.pickle"

MAX_SEQUENCE_LENGTH = 100

st.set_page_config(page_title="LSTM Sentiment Analyzer", layout="centered")

# Resource loader
@st.cache_resource
def load_resources():
    # load model
    model = load_model(str(MODEL_PATH))
    # load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    # load label encoder
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# App UI
st.title("LSTM Sentiment Analyzer")
st.write("Type (or paste) a tweet / review and click **Predict sentiment**.")

st.sidebar.header("Examples")
st.sidebar.write("- I love the airline service today!")
st.sidebar.write("- My flight was delayed and nobody helped 😡")
st.sidebar.write("- The crew was okay, nothing special.")

st.sidebar.markdown("---")
st.sidebar.caption("Model: Bidirectional LSTM trained for DSA6202 assignment")

# load resources with friendly error if files missing
try:
    model, tokenizer, le = load_resources()
except Exception as e:
    st.error("Model/resources could not be loaded. Make sure you ran `train.py` and that models/tokenizer exist.")
    st.exception(e)
    st.stop()


user_input = st.text_area("Text input", height=140, value="I love the airline service today!")
if st.button("Predict sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        with st.spinner("Predicting..."):
            try:
                probs = model.predict(pad, verbose=0)[0]
            except Exception as e:
                st.error("Prediction failed. See detail below.")
                st.exception(e)
                st.stop()

        pred_idx = int(np.argmax(probs))
        # inverse_transform expects an array of encoded labels
        try:
            label = le.inverse_transform([pred_idx])[0]
        except Exception:
            # fallback: map by classes_ order if label encoder doesn't match indices
            label = le.classes_[pred_idx]

        # display
        st.markdown("### Result")
        # highlight predicted label
        if label.lower() == "positive":
            st.success(f"Predicted sentiment: **{label}**")
        elif label.lower() == "negative":
            st.error(f"Predicted sentiment: **{label}**")
        else:
            st.info(f"Predicted sentiment: **{label}**")

        # pretty probabilities
        prob_map = {cls: float(probs[i]) for i, cls in enumerate(le.classes_)}
        # convert to percent and sort by highest first
        prob_display = {k: f"{v*100:.1f}%" for k, v in sorted(prob_map.items(), key=lambda x: -x[1])}
        st.write("**Probabilities:**")
        st.json(prob_display)

# footer / credit (assignment)
st.markdown("---")
st.caption("Done by @Ken — DSA6202: Unstructured Data Analytics & Applications (model deployment demo).")
