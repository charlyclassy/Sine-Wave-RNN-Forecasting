"""WaveSense AI Studio — SimpleRNN one-step-ahead sine-wave forecasting.

Designed to match train_rnn.py: np.arange(0, 300, 0.1), sin, 20 samples,
chronological 80/20 split, and a saved SimpleRNN(24) + Dense(1) model.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "sine_rnn.keras"
METRICS_PATH = OUT / "metrics.json"
PREDICTIONS_PATH = OUT / "predictions.csv"
TRAINING_CHART = OUT / "training_loss.png"
PREDICTIONS_CHART = OUT / "actual_vs_predicted.png"

# IMPORTANT: Match the training data generation precisely.
STEP = 0.1
STOP_TIME = 300.0
DEFAULT_WINDOW = 20
TRAIN_FRACTION = 0.8

st.set_page_config(
    page_title="WaveSense AI | RNN Forecasting",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0B1020 0%, #152747 100%);
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] { background: #162642 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F8FAFC !important; opacity: 1 !important; }
    [data-testid="stMetric"] {
        background: #203250;
        border: 1px solid #48607D;
        border-radius: 16px;
        padding: 18px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #F8FAFC !important;
        opacity: 1 !important;
    }
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label *,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"],
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSlider"] [data-testid="stTickBar"] *,
    [data-testid="stSlider"] p {
        color: #F8FAFC !important;
        opacity: 1 !important;
    }
    .wavesense-hero {
        background: linear-gradient(110deg, #284A90, #5635A4);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 25px;
    }
    .wavesense-hero h1 { font-size: 2.4rem; margin: 0 0 12px; color: white !important; }
    .wavesense-hero p { margin: 0; color: #E7EAFE !important; font-size: 1.04rem; }
    .stButton > button {
        color: white !important;
        background: linear-gradient(90deg, #6366F1, #7C3AED) !important;
        border-radius: 12px;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="wavesense-hero">
    <h1>🌊 WaveSense AI Studio</h1>
    <p>Time-series forecasting · TensorFlow / Keras SimpleRNN · 20-step input window</p>
    </div>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_saved_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return {
        col: np.asarray([float(row[col]) for row in rows], dtype=np.float64)
        for col in ("time", "actual", "predicted")
    }


@st.cache_resource(show_spinner="Loading trained RNN model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data(show_spinner=False)
def generate_training_wave():
    # Same computation as train_rnn.py, including the float32 cast after sin().
    times = np.arange(0, STOP_TIME, STEP)
    wave = np.sin(times).astype("float32")
    return times, wave


metrics = load_metrics()
window = int(metrics.get("window_size", DEFAULT_WINDOW))
times, wave = generate_training_wave()
num_sequences = len(wave) - window
train_sequence_count = int(num_sequences * TRAIN_FRACTION)
# Each sequence i predicts target i + window, so the first test target is:
first_test_index = train_sequence_count + window

with st.sidebar:
    st.header("⚙️ Project settings")
    st.write("**Task:** Predict the next sine-wave value")
    st.write(f"**Input:** {window} previous values")
    st.write(f"**Network:** SimpleRNN ({metrics.get('hidden_units', 24)} units) + Dense(1)")
    st.write("**Random seed:** 42")
    st.divider()
    view = st.radio(
        "Choose a view",
        ["Forecast results", "Interactive prediction", "Training results", "How RNNs work"],
    )

if view == "Forecast results":
    st.subheader("Actual vs predicted sine-wave values")
    saved = load_saved_predictions()
    if saved is None:
        st.warning("No test predictions found. Run `python train_rnn.py --epochs 12` in VS Code, then upload `outputs/predictions.csv` to GitHub.")
    else:
        actuals = saved["actual"]
        predicted = saved["predicted"]
        c1, c2, c3 = st.columns(3)
        # The saved metrics are the results of the actual training/evaluation run.
        displayed_mse = float(metrics.get("test_mse", np.mean((actuals - predicted) ** 2)))
        displayed_mae = float(metrics.get("test_mae", np.mean(np.abs(actuals - predicted))))
        c1.metric("Test mean squared error", f"{displayed_mse:.6f}")
        c2.metric("Test mean absolute error", f"{displayed_mae:.6f}")
        c3.metric("Test windows", f"{len(actuals):,}")

        max_points = len(actuals)
        if max_points > 1:
            n = st.slider(
                "Show most recent test points",
                min_value=1,
                max_value=max_points,
                value=min(250, max_points),
                step=1,
            )
        else:
            n = 1
        # Overlay is useful for overall shape, but the curves can nearly coincide.
        # The residual chart and zoomed view make small differences visible.
        shown_actual = actuals[-n:]
        shown_predicted = predicted[-n:]
        shown_time = saved["time"][-n:]
        residuals = shown_predicted - shown_actual

        st.markdown("#### 1. Actual vs predicted")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(shown_time, shown_actual, color="#1764AB", lw=2.4,
                label="Actual", zorder=2)
        ax.plot(shown_time, shown_predicted, color="#D94842", lw=1.7,
                ls="--", label="Predicted", zorder=3)
        ax.set(xlabel="Time", ylabel="Sine-wave value")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("The two lines overlap because the model's predictions are close to the actual values. Blue is actual; dashed red is predicted.")

        st.markdown("#### 2. Prediction error — difference made visible")
        fig, ax = plt.subplots(figsize=(11, 3.1))
        ax.plot(shown_time, residuals, color="#B93845", lw=1.8,
                label="Predicted minus actual")
        ax.fill_between(shown_time, 0, residuals, color="#D94842", alpha=0.18)
        ax.axhline(0, color="#334155", linestyle="--", lw=1)
        ax.set(xlabel="Time", ylabel="Prediction error")
        ax.grid(alpha=0.20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Zero means a perfect prediction. Values above zero are overpredictions; below zero are underpredictions. This graph has its own vertical scale to reveal small errors.")

        st.markdown("#### 3. Zoom in on individual forecasts")
        zoom_count = min(30, len(actuals))
        if zoom_count >= 2:
            default_start = max(0, len(actuals) - 30)
            zoom_start = st.slider("Choose the start of a 30-point close-up", 0,
                                   max(0, len(actuals) - zoom_count), default_start)
            sl = slice(zoom_start, zoom_start + zoom_count)
            x_zoom = saved["time"][sl]
            act_zoom = actuals[sl]
            pred_zoom = predicted[sl]
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(x_zoom, act_zoom, "o-", color="#1764AB", markersize=3.5,
                    lw=1.8, label="Actual")
            ax.plot(x_zoom, pred_zoom, "x--", color="#D94842", markersize=4,
                    lw=1.5, label="Predicted")
            ax.set(xlabel="Time", ylabel="Sine-wave value")
            ax.grid(alpha=0.20)
            ax.legend(loc="best")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.caption("Every forecast uses the 20 true previous points (one-step ahead). The error plot is zoomed vertically, not an indication of large errors.")
        if PREDICTIONS_CHART.exists():
            with st.expander("View saved full test chart"):
                st.image(str(PREDICTIONS_CHART), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=PREDICTIONS_PATH.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
        )

elif view == "Interactive prediction":
    st.subheader(f"Predict the next point from {window} previous points")
    st.write("Select a target point from the test region. The app uses **the same sine-wave samples and spacing as training**.")

    if not MODEL_PATH.exists():
        st.warning("Trained model missing: `outputs/sine_rnn.keras`. Upload the trained model from VS Code to GitHub.")
    else:
        idx = st.slider(
            "Choose target point in test region",
            min_value=first_test_index,
            max_value=len(wave) - 1,
            value=min(2500, len(wave) - 1),
            step=1,
        )
        previous = wave[idx - window:idx]  # exactly the model's training input
        actual = float(wave[idx])
        try:
            model = load_model()
            prediction = float(model.predict(previous.reshape(1, window, 1), verbose=0)[0, 0])
        except Exception as exc:
            st.error(f"Could not load or run the saved RNN model: {exc}")
            st.stop()

        absolute_error = abs(actual - prediction)
        a, b, c = st.columns(3)
        a.metric("Actual next value", f"{actual:.4f}")
        b.metric("Predicted next value", f"{prediction:.4f}")
        c.metric("Absolute error", f"{absolute_error:.4f}")

        # Show the 20 inputs plus both next-point values on the same coordinates.
        fig, ax = plt.subplots(figsize=(10, 4.6))
        past_times = times[idx - window:idx]
        target_time = times[idx]
        ax.plot(past_times, previous, "o-", color="#2962AB", label="Previous points", markersize=4)
        ax.plot([past_times[-1], target_time], [previous[-1], actual],
                color="#1A9A6C", linestyle="--", alpha=0.8)
        ax.scatter([target_time], [actual], s=100, marker="o", color="#1A9A6C", label="Actual next")
        ax.scatter([target_time], [prediction], s=130, marker="X", color="#D9534F", label="Predicted next")
        ax.set_title(f"One-step forecast at target index {idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Sine-wave value")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Check the very same test index against the saved prediction from training.
        saved = load_saved_predictions()
        saved_row = idx - first_test_index
        if saved is not None and 0 <= saved_row < len(saved["actual"]):
            recorded = float(saved["predicted"][saved_row])
            recorded_actual = float(saved["actual"][saved_row])
            agrees = (abs(recorded - prediction) < 0.002 and abs(recorded_actual - actual) < 0.002)
            with st.expander("Verify prediction against saved test results"):
                st.write(f"Recorded test prediction: **{recorded:.6f}**")
                st.write(f"Live model prediction: **{prediction:.6f}**")
                if agrees:
                    st.success("Live prediction matches the saved test-run prediction.")
                else:
                    st.warning("These predictions do not match. Check that `outputs/sine_rnn.keras` and `outputs/predictions.csv` were generated by the same training run.")

elif view == "Training results":
    st.subheader("RNN training and evaluation")
    if metrics:
        a, b, c = st.columns(3)
        a.metric("Training epochs", str(metrics.get("epochs", "—")))
        b.metric("Test MSE", f"{float(metrics['test_mse']):.6f}" if "test_mse" in metrics else "—")
        c.metric("Test MAE", f"{float(metrics['test_mae']):.6f}" if "test_mae" in metrics else "—")
    else:
        st.info("`outputs/metrics.json` is missing. Upload the file generated by `train_rnn.py`.")

    for path, heading in [
        (TRAINING_CHART, "Training and validation loss"),
        (PREDICTIONS_CHART, "Actual and predicted test values"),
    ]:
        st.markdown(f"#### {heading}")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"Missing `outputs/{path.name}`")
    if metrics:
        with st.expander("View all saved evaluation metrics"):
            st.json(metrics)

else:
    st.subheader("How a SimpleRNN predicts the next point")
    st.markdown(
        """
        **Sequence:** The 20 consecutive sine-wave values supplied to the model.

        **Hidden state:** A small internal memory that changes as the RNN reads each point.
        It helps the network track the wave's recent movement and direction.

        **One-step forecast:** After reading the 20 points, a Dense layer produces a
        single predicted value for the next point.

        **Reconstruction vs prediction:** This model forecasts the next number; it does not
        reconstruct images or classify handwritten digits.

        **Limitation:** Basic SimpleRNNs often struggle to learn patterns that depend on
        information far back in long sequences, partly because of vanishing gradients.
        LSTMs and GRUs can handle longer dependencies more effectively.

        **MSE and MAE:** These measure prediction error. Smaller values mean predictions
        are closer to the observed data on the evaluated sequences.
        """
    )
