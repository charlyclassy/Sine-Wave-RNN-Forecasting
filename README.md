# 🌊 WaveSense AI Studio — SimpleRNN Sine-Wave Forecasting

Beginner-friendly time-series forecasting project using **NumPy**, **TensorFlow/Keras `SimpleRNN`**, **Matplotlib** and **Streamlit**.

## 🎯 Goal

Predict the **next sine-wave value from the preceding 20 values**. The project uses an intentionally small `SimpleRNN(24) → Dense(1)` model, a fixed random seed of **42**, and an 80/20 chronological train/test split. The model predicts **one step ahead** using real previous values; it does not recursively forecast an entire future series.

## 🚀 Live Streamlit App

[**Launch WaveSense AI Studio**](https://sine-wave-rnn-forecasting-kvtodccxygwz87am37mnbx.streamlit.app/)

## 📂 Project files

```text
SineWave_RNN_VSCode_GitHub/
├── app.py                         # Streamlit web dashboard
├── train_rnn.py                   # Train and evaluate SimpleRNN
├── requirements.txt              # Python libraries
├── README.md
├── .streamlit/
│   └── config.toml                # Blue/dark Streamlit theme
└── outputs/                       # Generated after training
    ├── sine_rnn.keras
    ├── actual_vs_predicted.png
    ├── training_loss.png
    ├── predictions.csv
    └── metrics.json
```

## 🖥️ Run in VS Code (no virtual environment required)

1. Extract the project ZIP and open its folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Install dependencies into your existing Python installation:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Train for 12 epochs:

   ```powershell
   python train_rnn.py --epochs 12
   ```

5. Confirm that the five files were created in `outputs/`.
6. Launch the dashboard locally (optional):

   ```powershell
   python -m streamlit run app.py
   ```

Tested for Python syntax; training and predicted results are generated on your machine, not precomputed in this repository.

## ☁️ GitHub → Streamlit Community Cloud

1. Create a public GitHub repository such as `Sine-Wave-RNN-Forecasting`.
2. Upload `app.py`, `train_rnn.py`, `requirements.txt`, `README.md` and `.streamlit/config.toml`.
3. **After training**, upload the **actual generated** `outputs/` files. Don't create an empty `.keras` file, and don't rename an unrelated model to `sine_rnn.keras`.
4. Visit [Streamlit Community Cloud](https://share.streamlit.io/) and choose your GitHub repository, `main` branch and `app.py` as the entry point.
5. Deploy, then copy the live URL into the README link above.

**Note:** If the trained model isn't uploaded, the web app shows instructions rather than falsely claiming to have completed forecasting.

## 📈 Results

Your trained model achieved the following results on the held-out later sine-wave windows (as displayed by the deployed app):

| Metric | Result |
|---|---:|
| Test mean squared error (MSE) | 0.000025 |
| Test mean absolute error (MAE) | 0.004230 |
| Test windows | 596 |

Interactive examples after correcting the input spacing to **0.1**:

| Target index | Actual | Predicted | Absolute error |
|---|---:|---:|---:|
| 2500 | -0.9705 | -0.9742 | 0.0037 |
| 2600 | 0.6832 | 0.6748 | 0.0085 |

The **Forecast results** dashboard includes an overlaid comparison, an error plot (`predicted - actual`) that makes tiny differences visible, and a zoomed 30-point comparison. Results are for a synthetic sine wave, not a benchmark on real-world time series.


## 🧠 Explanation

- **Hidden state:** The RNN keeps a running summary of the sequence as it reads the 20 points, so the last prediction can depend on earlier values.
- **One limitation:** SimpleRNN may struggle with very long sequences because gradients can vanish. LSTM or GRU can work better for long-term patterns.
- **Loss:** Mean squared error compares predicted and real next values. Lower is better on the same evaluation set.

## ✅ Assignment checklist

- [x] Generate synthetic sine-wave data using NumPy
- [x] Prepare sequences of 20 historical points
- [x] Use TensorFlow/Keras `SimpleRNN`
- [x] Fix the random seed
- [x] Plot actual vs predicted values (after running training)
- [x] Explain hidden state and one RNN limitation

## Project scope

One-step-ahead prediction on a synthetic sine wave using the preceding 20 observations.
