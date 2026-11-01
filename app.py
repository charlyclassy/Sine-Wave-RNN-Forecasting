"""Streamlit dashboard for a trained next-step sine-wave SimpleRNN."""
import csv
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "sine_rnn.keras"
METRICS_PATH = OUT / "metrics.json"
PREDICTIONS_PATH = OUT / "predictions.csv"

st.set_page_config(page_title="WaveSense AI | RNN Forecasting", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0B1020,#142442); color:#F8FAFC;}
[data-testid="stSidebar"] {background:#162642 !important;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#F8FAFC !important;opacity:1 !important;}
[data-testid="stMetric"] {background:#1E304D; border:1px solid #435773; border-radius:14px;padding:15px;}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {color:#F8FAFC !important;opacity:1 !important;}
.hero {padding:27px 30px; border-radius:22px; background:linear-gradient(110deg,#27498D,#5637A5);margin-bottom:20px;}
.hero h1,.hero p{color:white !important;}
.stButton>button {background:linear-gradient(90deg,#6366F1,#7C3AED); color:white; border:0; border-radius:11px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class='hero'><h1>🌊 WaveSense AI Studio</h1>
<p>Time-series forecasting · TensorFlow / Keras SimpleRNN · 20-step input window</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Project settings")
    st.write("**Task:** Predict the next sine-wave value")
    st.write("**Input:** 20 previous values")
    st.write("**Network:** SimpleRNN (24 units) + Dense(1)")
    st.write("**Random seed:** 42")
    st.divider()
    view = st.radio("Choose a view", [
        "Forecast results", "Interactive prediction", "Training results", "How RNNs work"
    ])
    st.caption("Forecasting is different from classifying sequences or recognising speech.")

@st.cache_resource(show_spinner="Loading trained RNN...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_data
def read_predictions():
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in ("time", "actual", "predicted")}

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

if view == "Forecast results":
    st.subheader("Actual vs predicted sine-wave values")
    if not PREDICTIONS_PATH.exists():
        st.warning("Results not found. Run `python train_rnn.py --epochs 12` in VS Code, then upload the `outputs/` files to GitHub.")
    else:
        values = read_predictions()
        c1, c2, c3 = st.columns(3)
        c1.metric("Test mean squared error", f"{metrics['test_mse']:.6f}" if "test_mse" in metrics else "Not recorded")
        c2.metric("Test mean absolute error", f"{metrics['test_mae']:.6f}" if "test_mae" in metrics else "Not recorded")
        c3.metric("Test windows", f"{len(values['actual'])}")
        last_n = st.slider("Show most recent test points", min_value=50, max_value=min(600,len(values['actual'])), value=min(250,len(values['actual'])), step=10)
        chart = {
            "Actual": values["actual"][-last_n:],
            "Predicted": values["predicted"][-last_n:]
        }
        st.line_chart(chart)
        st.caption("Each prediction uses the 20 true previous points. This is one-step-ahead forecasting, not a recursive long-range forecast.")
        if (OUT / "actual_vs_predicted.png").exists():
            with st.expander("View saved experiment chart"):
                st.image(str(OUT / "actual_vs_predicted.png"), use_container_width=True)

elif view == "Interactive prediction":
    st.subheader("Predict the next point from 20 previous points")
    if not MODEL_PATH.exists():
        st.warning("Trained model missing: `outputs/sine_rnn.keras`. Train in VS Code and upload the actual model file to GitHub.")
    else:
        st.write("Move the slider to choose a point on the same synthetic sine-wave pattern used for training.")
        step = float(metrics.get("step", 0.08))
        total = int(metrics.get("total_points", 3000))
        win = int(metrics.get("window_size", 20))
        times = np.arange(total, dtype=np.float32) * step
        wave = np.sin(times).astype(np.float32)
        start_idx = int(total * float(metrics.get("train_fraction", 0.8)))
        idx = st.slider("Choose target point in test region", min_value=max(win,start_idx), max_value=total-1, value=min(start_idx+50,total-1))
        prior = wave[idx-win:idx]
        try:
            model = load_model(str(MODEL_PATH))
            predicted = float(model.predict(prior.reshape(1,win,1), verbose=0).reshape(-1)[0])
        except Exception as exc:
            st.error(f"Unable to load or run the trained model: {exc}")
            st.stop()
        actual = float(wave[idx])
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual next value", f"{actual:.4f}")
        c2.metric("Predicted next value", f"{predicted:.4f}")
        c3.metric("Absolute error", f"{abs(actual-predicted):.4f}")
        st.line_chart({"Previous 20 points": list(prior)+[np.nan], "Actual next": [np.nan]*win+[actual], "Predicted next": [np.nan]*win+[predicted]})
        

elif view == "Training results":
    st.subheader("Model training and evaluation")
    if metrics:
        st.json(metrics, expanded=False)
    else:
        st.info("No metrics saved yet. Run `train_rnn.py` first.")
    for filename, caption in [
        ("training_loss.png", "Training and validation loss"),
        ("actual_vs_predicted.png", "Actual versus predicted on test data"),
    ]:
        path = OUT / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Missing `{path.relative_to(ROOT)}`")

else:
    st.subheader("How the SimpleRNN works")
    st.markdown("""
**Sequence:** A list of values in time order. Here the input is the last **20** sine-wave points.

**Hidden state:** The RNN carries a small summary of previous inputs as it reads each point. It updates that summary at every time step, helping it recognise the wave's direction and pattern.

**Prediction:** After processing 20 points, a Dense layer converts the RNN output into one number: the estimated next sine-wave value.

**Limitation:** SimpleRNNs can struggle to remember information from very long sequences because gradients may vanish during training. LSTM and GRU networks often handle longer dependencies better.

**Loss:** Mean squared error (MSE) measures how far predictions are from actual values. A lower MSE indicates closer predictions for the evaluated data.

This project demonstrates **time-series forecasting**, not speech recognition or text classification, although RNNs can also be applied to sequence-based tasks.
    """)
