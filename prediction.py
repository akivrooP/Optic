"""
traffic_prediction_project.py

End-to-end network traffic prediction pipeline:
- Load dataset (CSV with 'timestamp' and 'total_bytes' columns) OR generate synthetic data
- Preprocess, resample, scale
- Create sliding windows for supervised learning
- Train LSTM model, evaluate, save
- Plot results
- Run a small Streamlit dashboard for live / one-step-ahead prediction

Usage:
    python traffic_prediction_project.py         # runs training + plots
    streamlit run traffic_prediction_project.py  # runs Streamlit dashboard (press 'q' to quit in terminal)

If you want only training or only dashboard, set the RUN_* flags below.
"""

import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import kagglehub

# -------------------------
# Configuration
# -------------------------
path = kagglehub.dataset_download("ravikumargattu/network-traffic-dataset")

DATA_CSV = "C:\\Users\\venka\\.cache\\kagglehub\\datasets\\ravikumargattu\\network-traffic-dataset\\versions\\2\\Midterm_53_group.csv"     # expected columns: timestamp, total_bytes (or total_traffic)
TIMESTAMP_COL = "Time"
VALUE_COL = "Length"    # change if your CSV uses different name

RESAMPLE_RULE = "1Min"       # resample frequency for aggregation: "1Min", "30S", "5Min", etc.
WINDOW_SIZE = 30             # number of past steps to use as input (e.g., past 30 minutes)
PREDICT_AHEAD = 1            # predict next 1 interval (1 minute if RESAMPLE_RULE="1Min")
TEST_RATIO = 0.2
RANDOM_SEED = 42
MODEL_DIR = "saved_model"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")
EPOCHS = 30
BATCH_SIZE = 32

# Flags to run steps
RUN_TRAIN = True
RUN_PLOT = True
RUN_SAVE = True
RUN_STREAMLIT = False   # set to True if you want to run dashboard when running script directly

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# Helper functions
# -------------------------
def generate_synthetic_traffic(start_time="2025-01-01 00:00:00", minutes=24*60, seed=RANDOM_SEED):
    """Create synthetic traffic time series for testing (diurnal pattern + noise + random spikes)."""
    rng = pd.date_range(start=start_time, periods=minutes, freq="1Min")
    np.random.seed(seed)
    base = 2000 + 800 * (np.sin(np.linspace(0, 2 * np.pi, minutes)) )  # daily sinusoidal
    trend = np.linspace(0, 500, minutes)  # slight upward trend over the day
    noise = np.random.normal(0, 150, minutes)
    spikes = np.zeros(minutes)
    # random spikes (simulate bursts)
    for _ in range(int(minutes / 200)):
        idx = np.random.randint(0, minutes-10)
        spikes[idx:idx+5] += np.random.randint(2000, 8000)
    total = np.clip(base + trend + noise + spikes, a_min=0, a_max=None)
    df = pd.DataFrame({TIMESTAMP_COL: rng, VALUE_COL: total})
    return df

def load_or_generate_data(csv_path=DATA_CSV):
    """Load CSV if present; otherwise generate synthetic data. Returns DataFrame with datetime index."""
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        if TIMESTAMP_COL not in df.columns or VALUE_COL not in df.columns:
            raise ValueError(f"CSV must contain columns: {TIMESTAMP_COL}, {VALUE_COL}")
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df = df.set_index(TIMESTAMP_COL).sort_index()
    else:
        print("CSV not found. Generating synthetic dataset (for demo).")
        df = generate_synthetic_traffic()
        df = df.set_index(TIMESTAMP_COL)
    return df

def resample_and_fill(df, rule=RESAMPLE_RULE):
    """Resample to fixed frequency and fill missing values via forward/backward fill."""
    s = df[VALUE_COL].resample(rule).sum()  # or .mean() based on your data semantics
    s = s.ffill().bfill()
    return s.to_frame(name=VALUE_COL)

def scale_series(df, scaler=None):
    """Scale series to [0,1] using MinMaxScaler. If scaler provided, uses it; otherwise fits a new one."""
    arr = df[[VALUE_COL]].values.astype("float32")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        arr_scaled = scaler.fit_transform(arr)
    else:
        arr_scaled = scaler.transform(arr)
    df_scaled = pd.DataFrame(arr_scaled, index=df.index, columns=[VALUE_COL])
    return df_scaled, scaler

def create_windows(series_array, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD):
    """
    Convert series array (N,1) to windows:
    X -> (num_samples, window_size, 1)
    y -> (num_samples, predict_ahead)
    """
    X, y = [], []
    N = len(series_array)
    for i in range(N - window_size - predict_ahead + 1):
        X.append(series_array[i:i + window_size])
        y.append(series_array[i + window_size:i + window_size + predict_ahead])
    X = np.array(X)
    y = np.array(y)
    # if predict_ahead==1, squeeze last dim
    if y.shape[-1] == 1:
        y = y.reshape((y.shape[0], 1))
    return X, y

def train_test_split_timewise(X, y, test_ratio=TEST_RATIO):
    """Split arrays by time (no shuffling)."""
    n = len(X)
    split = int(n * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# -------------------------
# Model builders
# -------------------------
def build_lstm_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(PREDICT_AHEAD)   # regression output
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

def build_cnn1d_model(input_shape):
    # optional alternative baseline
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(16, kernel_size=3, activation="relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(PREDICT_AHEAD)
    ])
    model.compile(optimizer='adam', loss="mse", metrics=["mae"])
    return model

# -------------------------
# Training pipeline
# -------------------------
def train_pipeline():
    # 1) Load
    df_raw = load_or_generate_data()
    # 2) Resample & clean
    df_resampled = resample_and_fill(df_raw)
    print(f"Resampled series length: {len(df_resampled)} ({df_resampled.index[0]} -> {df_resampled.index[-1]})")
    # 3) Scale
    df_scaled, scaler = scale_series(df_resampled)
    # 4) Windowing
    arr = df_scaled.values  # shape (N,1)
    X, y = create_windows(arr, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD)
    print(f"Created windows -> X: {X.shape}, y: {y.shape}")
    # 5) Train-test split
    X_train, X_test, y_train, y_test = train_test_split_timewise(X, y)
    # 6) Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, 1)
    model = build_lstm_model(input_shape)
    model.summary()
    # 7) Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    # 8) Evaluate on test
    test_pred = model.predict(X_test)
    # inverse scale for meaningful metrics
    # we must inverse transform both predictions and y_test; they are shaped (n,1)
    pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
    ytest_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(ytest_inv, pred_inv))
    mae = mean_absolute_error(ytest_inv, pred_inv)
    print(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    # 9) Save model + scaler
    if RUN_SAVE:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")
    return {
        "model": model,
        "scaler": scaler,
        "df_resampled": df_resampled,
        "df_scaled": df_scaled,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "history": history
    }

# -------------------------
# Prediction & plotting utilities
# -------------------------
def plot_predictions(df_resampled, scaler, X_test, y_test, model, num_points=300):
    """Plot actual vs predicted for the last num_points samples of test set."""
    pred = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    ytest_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Timestamps for test ground truth: compute from df_resampled index
    # The windows produce targets starting at index window_size ... so find corresponding time indexes
    all_indices = df_resampled.index
    start_idx = WINDOW_SIZE + (len(X_test) * 0)  # starting point for last segment is split marker; we'll reconstruct below

    # For easy plotting, we construct arrays aligned by sample index (not exact timestamps)
    last_n = min(num_points, len(pred_inv))
    plt.figure(figsize=(12,5))
    plt.plot(range(last_n), ytest_inv[-last_n:], label="Actual")
    plt.plot(range(last_n), pred_inv[-last_n:], label="Predicted")
    plt.title("Actual vs Predicted (last {} test points)".format(last_n))
    plt.xlabel("Sample index (test set)")
    plt.ylabel("Traffic (original units)")
    plt.legend()
    plt.grid(True)
    plt.show()

def one_step_predict_recent(model, scaler, df_resampled, window_size=WINDOW_SIZE):
    """Take the last window_size values and predict next interval; returns value in original scale."""
    last_window = df_resampled[VALUE_COL].iloc[-window_size:].values.reshape(-1,1).astype("float32")
    # scale
    last_window_scaled = scaler.transform(last_window)
    X = last_window_scaled.reshape((1, window_size, 1))
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
    return pred

# -------------------------
# Main flow
# -------------------------
if __name__ == "__main__":
    if RUN_TRAIN:
        artifacts = train_pipeline()
    else:
        # If not training, load model/scaler if available
        artifacts = {}
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            print("Loading existing model and scaler...")
            artifacts["model"] = keras.models.load_model(MODEL_PATH)
            artifacts["scaler"] = joblib.load(SCALER_PATH)
            artifacts["df_resampled"] = resample_and_fill(load_or_generate_data())
        else:
            raise RuntimeError("No saved model/scaler found. Set RUN_TRAIN=True to train or put a model in saved_model/")

    if RUN_PLOT:
        plot_predictions(
            df_resampled=artifacts["df_resampled"],
            scaler=artifacts["scaler"],
            X_test=artifacts["X_test"],
            y_test=artifacts["y_test"],
            model=artifacts["model"],
            num_points=500
        )

    # -------------------------
    # Streamlit app (optional)
    # -------------------------
    # To run: streamlit run traffic_prediction_project.py
    if RUN_STREAMLIT:
        import streamlit as st
        st.title("Network Traffic Prediction (LSTM)")
        st.markdown("This demo predicts the next time interval's traffic using the last "
                    f"{WINDOW_SIZE} intervals (resample rule: {RESAMPLE_RULE}).")

        # Show recent data
        df_view = artifacts["df_resampled"].tail(200)
        st.line_chart(df_view[VALUE_COL])

        if st.button("Predict next interval (one-step)"):
            pred_val = one_step_predict_recent(artifacts["model"], artifacts["scaler"], artifacts["df_resampled"])
            st.metric(label="Predicted traffic (next interval)", value=f"{pred_val:.2f}")
            st.write("Prediction time:", pd.Timestamp.now())

        st.write("Upload CSV (optional) with columns 'timestamp' and 'total_bytes' to replace dataset")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df_u = pd.read_csv(uploaded)
            df_u[TIMESTAMP_COL] = pd.to_datetime(df_u[TIMESTAMP_COL])
            df_u = df_u.set_index(TIMESTAMP_COL).sort_index()
            df_res = resample_and_fill(df_u)
            st.write("Preview:")
            st.dataframe(df_res.tail(200))
            if st.button("Predict from uploaded data"):
                pred = one_step_predict_recent(artifacts["model"], artifacts["scaler"], df_res)
                st.metric("Predicted next interval (uploaded data)", f"{pred:.2f}")

