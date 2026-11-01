
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential


# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12)
args = parser.parse_args()

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

WINDOW = 20
STEP = 0.1

# 1. Generate sine-wave data
print("Generating sine-wave data...")

time = np.arange(0, 300, STEP)
values = np.sin(time).astype("float32")

# 2. Prepare sequences of 20 points
X = []
y = []
target_times = []

for i in range(len(values) - WINDOW):
    X.append(values[i:i + WINDOW])
    y.append(values[i + WINDOW])
    target_times.append(time[i + WINDOW])

X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")
target_times = np.array(target_times)

# SimpleRNN expects (samples, timesteps, features)
X = X.reshape(-1, WINDOW, 1)

# 3. Split data chronologically
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

test_times = target_times[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input shape: {X_train.shape}")

# 4. Build the SimpleRNN model
print("Building SimpleRNN model...")

model = Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.SimpleRNN(24, activation="tanh"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# 5. Train the model
print(f"Training for {args.epochs} epochs...")

history = model.fit(
    X_train,
    y_train,
    epochs=args.epochs,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)

# 6. Evaluate on unseen later points
print("Evaluating model...")

test_mse, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test MSE: {test_mse:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# 7. Predict next sine-wave values
predictions = model.predict(
    X_test,
    verbose=0
).flatten()

# 8. Save the trained model
model.save(OUTPUT / "sine_rnn.keras")

# 9. Plot actual versus predicted values
plt.figure(figsize=(12, 5))

plt.plot(
    test_times,
    y_test,
    label="Actual sine wave",
    linewidth=2
)

plt.plot(
    test_times,
    predictions,
    label="Predicted sine wave",
    linestyle="--",
    linewidth=1.5
)

plt.title("SimpleRNN: Actual vs Predicted Sine Wave")
plt.xlabel("Time")
plt.ylabel("Sine-wave value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "actual_vs_predicted.png",
    dpi=180
)
plt.close()

# 10. Plot training loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss"
)

plt.title("SimpleRNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "training_loss.png",
    dpi=180
)
plt.close()

# 11. Save predictions to CSV
with open(
    OUTPUT / "predictions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(
        ["time", "actual", "predicted"]
    )

    for t, actual, predicted in zip(
        test_times, y_test, predictions
    ):
        writer.writerow([
            float(t),
            float(actual),
            float(predicted)
        ])

# 12. Save evaluation results
metrics = {
    "model": "SimpleRNN",
    "window_size": WINDOW,
    "hidden_units": 24,
    "epochs": args.epochs,
    "test_mse": float(test_mse),
    "test_mae": float(test_mae)
}

with open(
    OUTPUT / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nTraining completed successfully!")
print("Files saved inside outputs/:")
print(" - sine_rnn.keras")
print(" - actual_vs_predicted.png")
print(" - training_loss.png")
print(" - predictions.csv")
print(" - metrics.json")
