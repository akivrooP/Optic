"""
prediction.py

Robust end-to-end network traffic prediction script.

Features:
- Loads CSV (Wireshark-style or other). Auto-detects timestamp and traffic columns.
- Converts "Time" (seconds offset) into DatetimeIndex or parses epoch/ISO timestamps.
- Resamples to fixed time bins (default 1 second) and aggregates packet length by sum.
- Creates sliding windows for supervised learning.
- Trains an LSTM regressor and evaluates (RMSE, MAE).
- Saves model and scaler.
- Optional Streamlit dashboard for one-step prediction.

Edit configuration near the top to adapt frequency, window size, model params, and file path.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

# -------------------------
# Configuration (edit here)
# -------------------------
  # path to your CSV; change if needed
DATA_CSV = "C:\\Users\\venka\\.cache\\kagglehub\\datasets\\ravikumargattu\\network-traffic-dataset\\versions\\2\\Midterm_53_group.csv"     # expected columns: timestamp, total_bytes (or total_traffic)
AUTO_DETECT_COLUMNS = True    # try to auto-detect timestamp/value columns
TIMESTAMP_COL_PREFERRED = "Time"   # preferred timestamp column name (seconds offset in your file)
VALUE_COL_PREFERRED = "Length"     # preferred traffic/value column (packet length)
RESAMPLE_RULE = "1Min"          # "1S" for 1-second bins, "1Min" for 1-minute bins, etc.
WINDOW_SIZE = 10              # number of past timesteps used as input
PREDICT_AHEAD = 1             # how many steps ahead to predict
TEST_RATIO = 0.2
RANDOM_SEED = 42
MODEL_DIR = "saved_model"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")

EPOCHS = 30
BATCH_SIZE = 32

RUN_TRAIN = True
RUN_PLOT = True
RUN_SAVE = True
RUN_STREAMLIT = False   # set True to enable streamlit part (run via `streamlit run prediction.py`)

# deterministic-ish
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# Utility & Robust Loader
# -------------------------
def debug_print(msg):
    print(msg)

def safe_strip_quotes(s):
    if isinstance(s, str):
        return s.strip().strip('"').strip("'")
    return s

def load_csv_as_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    # read with default settings; we'll clean names
    df = pd.read_csv(path, low_memory=False)
    # clean column names
    df.columns = [safe_strip_quotes(c).strip() for c in df.columns]
    return df

def try_parse_timestamp_series(series):
    """Try multiple strategies to parse a series into datetimes. Returns pd.DatetimeIndex or None."""
    # 1) direct to_datetime
    try:
        dt = pd.to_datetime(series, errors='raise', infer_datetime_format=True)
        return dt
    except Exception:
        pass
    # 2) numeric -> seconds since start (if small numbers)
    # allow floats or ints
    try:
        s_numeric = pd.to_numeric(series, errors='coerce')
        if s_numeric.notna().sum() > 0:
            # if values look like small offsets (max < 1e7), treat as seconds offsets
            if s_numeric.max() < 1e8:
                base = pd.Timestamp.now().normalize()  # arbitrary base
                dt = base + pd.to_timedelta(s_numeric - s_numeric.min(), unit='s')
                return dt
            # if values look like epoch seconds (> 1e9), treat as epoch seconds
            if s_numeric.max() > 1e9:
                dt = pd.to_datetime(s_numeric, unit='s', errors='coerce')
                if dt.notna().sum() > 0:
                    return dt
    except Exception:
        pass
    # 3) try infer with coerce
    try:
        dt = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
        if dt.notna().sum() > 0:
            return dt
    except Exception:
        pass
    return None

def detect_columns_and_prepare(df):
    """
    Returns DataFrame with DatetimeIndex and a single column VALUE_COL.
    Strategy:
    - If preferred timestamp column exists, attempt to parse it (handles seconds offsets).
    - Else auto-detect candidate timestamp columns.
    - Value column: prefer preferred value column else pick first numeric column.
    """
    df = df.copy()
    debug_print("CSV columns: " + ", ".join(df.columns.tolist()))
    # make string columns uniform
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).apply(lambda x: safe_strip_quotes(x))
    # Attempt preferred timestamp parsing
    dt_index = None
    ts_col_used = None
    if TIMESTAMP_COL_PREFERRED in df.columns:
        debug_print(f"Found preferred timestamp column '{TIMESTAMP_COL_PREFERRED}'. Trying to parse it.")
        dt_try = try_parse_timestamp_series(df[TIMESTAMP_COL_PREFERRED])
        if dt_try is not None and dt_try.notna().sum() >= 2:
            dt_index = dt_try
            ts_col_used = TIMESTAMP_COL_PREFERRED
    # Auto-detect if not found
    if dt_index is None:
        # collect candidate names
        candidates = []
        for c in df.columns:
            name_lower = c.lower()
            if any(k in name_lower for k in ["time", "date", "timestamp", "ts"]):
                candidates.append(c)
        debug_print("Timestamp candidates by name: " + ", ".join(candidates))
        for c in candidates:
            dt_try = try_parse_timestamp_series(df[c])
            if dt_try is not None and dt_try.notna().sum() >= 2:
                dt_index = dt_try
                ts_col_used = c
                debug_print(f"Auto-detected timestamp column: {c}")
                break
    # If still not found, try numeric columns as offsets
    if dt_index is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            dt_try = try_parse_timestamp_series(df[c])
            if dt_try is not None and dt_try.notna().sum() >= 2:
                dt_index = dt_try
                ts_col_used = c
                debug_print(f"Using numeric column '{c}' as timestamp offsets (auto-detected).")
                break
    # If still none, fallback: create synthetic increasing index
    if dt_index is None:
        debug_print("No timestamp column could be parsed reliably. Falling back to synthetic index (1s spacing).")
        n = len(df)
        dt_index = pd.date_range(start=pd.Timestamp.now().normalize(), periods=n, freq='1S')
        ts_col_used = None

    # Ensure VALUE column present
    if VALUE_COL_PREFERRED in df.columns:
        val_col = VALUE_COL_PREFERRED
    else:
        # pick first numeric column that is not the timestamp column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != ts_col_used]
        if len(numeric_cols) == 0:
            # try columns that look numeric but are strings
            potential = []
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace('"',''), errors='raise')
                    potential.append(c)
                except Exception:
                    pass
            potential = [c for c in potential if c != ts_col_used]
            if not potential:
                raise ValueError("No numeric/value column found in CSV to use as traffic measurement.")
            val_col = potential[0]
        else:
            val_col = numeric_cols[0]
            debug_print(f"VALUE column '{VALUE_COL_PREFERRED}' not found – using '{val_col}' as value column.")

    # Build final DataFrame with DatetimeIndex and the value column
    df_out = pd.DataFrame(index=pd.DatetimeIndex(dt_index))
    # coerce value column to numeric
    df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace('"',''), errors='coerce').fillna(0).astype(float)
    df_out[VALUE_COL_PREFERRED] = df[val_col].values
    df_out.index.name = "datetime"
    debug_print(f"Data prepared: {len(df_out)} rows; time range {df_out.index[0]} -> {df_out.index[-1]}")
    return df_out

def load_data(path):
    df = load_csv_as_df(path)
    df_prepared = detect_columns_and_prepare(df)
    return df_prepared

# -------------------------
# Resample / windowing / scaler
# -------------------------
def resample_and_fill(df, rule=RESAMPLE_RULE):
    """Resample the series to fixed frequency and aggregate by sum (good for packet lengths)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for resampling.")
    if len(df) == 0:
        raise ValueError("Empty DataFrame passed to resample_and_fill.")
    if rule is None:
        s = df.iloc[:,0].ffill().bfill()
        return s.to_frame(name=df.columns[0])
    # try resample
    try:
        s = df.iloc[:,0].resample(rule).sum()
        s = s.fillna(0)
        return s.to_frame(name=df.columns[0])
    except Exception as e:
        debug_print(f"Resample failed ({e}); returning original series filled.")
        s = df.iloc[:,0].ffill().bfill()
        return s.to_frame(name=df.columns[0])

def scale_series(df, scaler=None):
    arr = df.values.astype("float32")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

def create_windows(series_array, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD):
    X, y = [], []
    N = len(series_array)
    for i in range(N - window_size - predict_ahead + 1):
        X.append(series_array[i:i + window_size])
        y.append(series_array[i + window_size:i + window_size + predict_ahead])
    X = np.array(X)
    y = np.array(y)
    if y.size == 0:
        return X, y
    if y.shape[-1] == 1:
        y = y.reshape((y.shape[0], 1))
    return X, y

def timewise_train_test_split(X, y, test_ratio=TEST_RATIO):
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]

# -------------------------
# Model architectures
# -------------------------
def build_lstm_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(PREDICT_AHEAD)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# -------------------------
# Train / evaluate pipeline
# -------------------------
def train_pipeline(csv_path=DATA_CSV):
    # load
    df_loaded = load_data(csv_path)
    # resample
    df_resampled = resample_and_fill(df_loaded, rule=RESAMPLE_RULE)
    debug_print(f"After resampling: {len(df_resampled)} rows; index {df_resampled.index[0]} -> {df_resampled.index[-1]}")
    # validate enough samples
    min_required = WINDOW_SIZE + PREDICT_AHEAD + 1
    if len(df_resampled) < min_required:
        raise ValueError(
            f"Not enough samples after resampling ({len(df_resampled)}). Need at least {min_required} "
            f"for windowing with WINDOW_SIZE={WINDOW_SIZE} and PREDICT_AHEAD={PREDICT_AHEAD}. "
            "Either reduce WINDOW_SIZE or resample to coarser interval (e.g., '1Min')."
        )

    # scale
    df_scaled, scaler = scale_series(df_resampled)
    arr = df_scaled.values  # shape (N,1)
    # windowing
    X, y = create_windows(arr, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD)
    if X.size == 0:
        raise ValueError("Window creation returned zero samples. Check WINDOW_SIZE and resampled data length.")
    debug_print(f"Created windows: X shape {X.shape}, y shape {y.shape}")

    # reshape X to (samples, window_size, 1)
    if X.ndim == 3:
        pass
    elif X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    else:
        X = X.reshape((X.shape[0], WINDOW_SIZE, 1))

    X_train, X_test, y_train, y_test = timewise_train_test_split(X, y)
    debug_print(f"Train/Test sizes: X_train {X_train.shape}, X_test {X_test.shape}")

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )

    # evaluate
    pred_scaled = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    ytest_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(ytest_inv, pred_inv))
    mae = mean_absolute_error(ytest_inv, pred_inv)
    debug_print(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    # save
    if RUN_SAVE:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        debug_print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

    artifacts = dict(
        model=model,
        scaler=scaler,
        df_resampled=df_resampled,
        df_scaled=df_scaled,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        history=history
    )
    return artifacts

# -------------------------
# Plot utilities
# -------------------------
def plot_predictions(df_resampled, scaler, X_test, y_test, model, num_points=300):
    pred_scaled = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    ytest_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    n = min(num_points, len(pred_inv))
    plt.figure(figsize=(12,5))
    plt.plot(range(n), ytest_inv[-n:], label='Actual')
    plt.plot(range(n), pred_inv[-n:], label='Predicted')
    plt.title(f'Actual vs Predicted (last {n} test points)')
    plt.xlabel('Sample index (test set)')
    plt.ylabel('Traffic (original units)')
    plt.legend()
    plt.grid(True)
    plt.show()

def one_step_predict_recent(model, scaler, df_resampled, window_size=WINDOW_SIZE):
    last_window = df_resampled.iloc[-window_size:].values.astype("float32")
    last_window_scaled = scaler.transform(last_window)
    X = last_window_scaled.reshape((1, window_size, 1))
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    return pred

# -------------------------
# Main
# -------------------------
def main(args):
    global DATA_CSV, RESAMPLE_RULE, WINDOW_SIZE, PREDICT_AHEAD
    if args.csv:
        DATA_CSV = args.csv
    if args.resample:
        RESAMPLE_RULE = args.resample
    if args.window:
        WINDOW_SIZE = int(args.window)
    if args.epochs:
        global EPOCHS
        EPOCHS = int(args.epochs)

    if RUN_TRAIN:
        artifacts = train_pipeline(DATA_CSV)
    else:
        # attempt to load saved artifacts
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            artifacts = {}
            artifacts['model'] = keras.models.load_model(MODEL_PATH)
            artifacts['scaler'] = joblib.load(SCALER_PATH)
            # load last data for plotting/predicting
            artifacts['df_resampled'] = resample_and_fill(load_data(DATA_CSV), rule=RESAMPLE_RULE)
        else:
            raise RuntimeError("No saved model/scaler found. Set RUN_TRAIN=True or provide saved model files.")

    if RUN_PLOT:
        plot_predictions(
            df_resampled=artifacts['df_resampled'],
            scaler=artifacts['scaler'],
            X_test=artifacts['X_test'],
            y_test=artifacts['y_test'],
            model=artifacts['model'],
            num_points=500
        )

    # optional streamlit UI (run via `streamlit run prediction.py`)
    if RUN_STREAMLIT:
        try:
            import streamlit as st
            st.title("Network Traffic Prediction (LSTM)")
            st.write("Resample rule:", RESAMPLE_RULE, "Window size:", WINDOW_SIZE)
            st.line_chart(artifacts['df_resampled'].tail(200))
            if st.button("Predict next interval"):
                val = one_step_predict_recent(artifacts['model'], artifacts['scaler'], artifacts['df_resampled'])
                st.metric("Predicted next interval", f"{val:.2f}")
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is not None:
                df_new = load_csv_as_df(uploaded)
                df_p = detect_columns_and_prepare(df_new)
                df_res = resample_and_fill(df_p, rule=RESAMPLE_RULE)
                st.line_chart(df_res.tail(200))
                if st.button("Predict (uploaded)"):
                    pred = one_step_predict_recent(artifacts['model'], artifacts['scaler'], df_res)
                    st.metric("Predicted", f"{pred:.2f}")
        except Exception as e:
            debug_print("Streamlit section failed to run: " + str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network traffic prediction")
    parser.add_argument("--csv", help="Path to CSV file", default=None)
    parser.add_argument("--resample", help="Resample rule (e.g., '1S', '1Min')", default=None)
    parser.add_argument("--window", help="Window size (int)", default=None)
    parser.add_argument("--epochs", help="Training epochs", default=None)
    args = parser.parse_args()
    main(args)
