# prediction_final.py
"""
Robust network traffic prediction (final).
- Handles Wireshark-style CSV where "Time" is seconds-offset floats.
- Resamples to 1-second bins, windowing, trains LSTM, evaluates, saves model/scaler, and plots.
"""

import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

# ---------------- CONFIG (edit if needed) ----------------
DEFAULT_CSV = r"C:\Users\venka\.cache\kagglehub\datasets\ravikumargattu\network-traffic-dataset\versions\2\Midterm_53_group.csv"
TIMESTAMP_COL = "Time"      # seconds-offset column in your file
VALUE_COL = "Length"        # packet length column
RESAMPLE_RULE = "1s"        # 1-second bins (lowercase 's' to avoid pandas warning)
WINDOW_SIZE = 30            # history length (in resampled units)
PREDICT_AHEAD = 1
TEST_RATIO = 0.2
RANDOM_SEED = 42
MODEL_DIR = "saved_model"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")
EPOCHS = 20
BATCH_SIZE = 32
AUTO_ADJUST_WINDOW = True   # if dataset too small, reduce window automatically

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------- Helpers ----------------
def debug(msg):
    print(msg)

def safe_strip(s):
    return s.strip().strip('"').strip("'") if isinstance(s, str) else s

# ---------------- Load & prepare ----------------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [safe_strip(c).strip() for c in df.columns]
    return df

def prepare_dataframe(df):
    # sanitize object columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).apply(safe_strip)
    # handle Time as seconds-offset floats (preferred)
    if TIMESTAMP_COL in df.columns:
        debug(f"Parsing '{TIMESTAMP_COL}' as numeric seconds-offsets.")
        times = pd.to_numeric(df[TIMESTAMP_COL].astype(str).str.replace('"',''), errors='coerce')
        if times.isna().all():
            raise ValueError(f"'{TIMESTAMP_COL}' could not be parsed as numeric offsets.")
        base = pd.Timestamp.now().normalize()
        offsets = times.fillna(0) - times.min()
        dt_index = base + pd.to_timedelta(offsets, unit='s')
    else:
        # fallback: try to find a time-like column
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("time","date","timestamp","ts"))]
        parsed = None
        for c in candidates:
            parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            if parsed.notna().sum() >= 2:
                dt_index = parsed
                debug(f"Auto-detected timestamp column '{c}'.")
                break
        if parsed is None or parsed.notna().sum() < 2:
            debug("No timestamp found — building synthetic 1s-spaced index.")
            dt_index = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(df), freq='1s')

    # choose value column
    if VALUE_COL in df.columns:
        val_col = VALUE_COL
    else:
        # pick first numeric column
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            # try coercion
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace('"',''), errors='raise')
                    numeric.append(c)
                except Exception:
                    pass
        if not numeric:
            raise ValueError("No numeric/value column found to use as traffic measurement.")
        val_col = numeric[0]
        debug(f"Using column '{val_col}' as value column.")

    # build final dataframe
    df_out = pd.DataFrame(index=pd.DatetimeIndex(dt_index))
    df_out.index.name = "datetime"
    df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace('"',''), errors='coerce').fillna(0).astype(float)
    df_out[VALUE_COL] = df[val_col].values
    debug(f"Prepared data: rows={len(df_out)}, time {df_out.index[0]} -> {df_out.index[-1]}")
    return df_out

# ---------------- Resample / scale / windows ----------------
def resample_series(df, rule=RESAMPLE_RULE):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex to resample.")
    if rule is None:
        return df.fillna(0)
    s = df[VALUE_COL].resample(rule).sum().fillna(0)
    debug(f"Resampled to {rule}: {len(s)} rows ({s.index[0]} -> {s.index[-1]})")
    return s.to_frame(name=VALUE_COL)

def scale(df, scaler=None):
    arr = df.values.astype("float32")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

def make_windows(arr, window, predict):
    X, y = [], []
    N = len(arr)
    for i in range(N - window - predict + 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window:i+window+predict])
    X = np.array(X)
    y = np.array(y)
    if y.size == 0:
        return X, y
    if y.shape[-1] == 1:
        y = y.reshape((y.shape[0], 1))
    return X, y

# ---------------- Model ----------------
def build_model(input_shape):
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

# ---------------- Pipeline ----------------
def run_pipeline(csv_path):
    df_raw = load_csv(csv_path)
    df_prepared = prepare_dataframe(df_raw)
    df_resampled = resample_series(df_prepared, rule=RESAMPLE_RULE)

    # auto-adjust window if needed
    global WINDOW_SIZE
    min_needed = WINDOW_SIZE + PREDICT_AHEAD + 1
    if len(df_resampled) < min_needed:
        if AUTO_ADJUST_WINDOW and len(df_resampled) > (PREDICT_AHEAD + 1):
            old = WINDOW_SIZE
            WINDOW_SIZE = max(1, len(df_resampled) - PREDICT_AHEAD - 1)
            debug(f"Not enough samples ({len(df_resampled)}). Auto-adjusted WINDOW_SIZE {old} -> {WINDOW_SIZE}.")
        else:
            raise ValueError(f"Not enough samples after resampling ({len(df_resampled)}). Need >= {min_needed}.")

    df_scaled, scaler = scale(df_resampled)
    arr = df_scaled.values
    X, y = make_windows(arr, WINDOW_SIZE, PREDICT_AHEAD)
    if X.size == 0:
        raise RuntimeError("Windowing returned zero samples; check parameters.")
    debug(f"Windows: X {X.shape}, y {y.shape}")

    # reshape
    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    debug(f"Train/Test sizes: {X_train.shape}, {X_test.shape}")

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    pred_scaled = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred_scaled.reshape(-1,1))
    ytest_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    rmse = math.sqrt(mean_squared_error(ytest_inv, pred_inv))
    mae = mean_absolute_error(ytest_inv, pred_inv)
    debug(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    # save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    debug(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

    # plot last portion
    try:
        n = min(500, len(pred_inv))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        plt.plot(range(n), ytest_inv.flatten()[-n:], label='Actual')
        plt.plot(range(n), pred_inv.flatten()[-n:], label='Predicted')
        plt.title('Actual vs Predicted (last {} test points)'.format(n))
        plt.xlabel('Test sample index')
        plt.ylabel('Traffic (original units)')
        plt.legend(); plt.grid(True); plt.show()
    except Exception as e:
        debug("Plotting failed: " + str(e))

    # also show one-step prediction for most recent window
    last_window = df_resampled.iloc[-WINDOW_SIZE:].values.astype('float32')
    last_scaled = scaler.transform(last_window)
    Xlast = last_scaled.reshape((1, WINDOW_SIZE, 1))
    pred_last_scaled = model.predict(Xlast)
    pred_last = scaler.inverse_transform(pred_last_scaled.reshape(-1,1)).flatten()[0]
    debug(f"One-step prediction (next interval): {pred_last:.2f}")

# ---------------- Entry ----------------
def load_csv(path):
    return load_csv_inner(path)

def load_csv_inner(path):
    return load_csv_original(path)

def load_csv_original(path):
    return load_csv_final(path)

def load_csv_final(path):
    # single loader to keep naming simple
    return load_csv_actual(path)

def load_csv_actual(path):
    # actual loader implementation
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [safe_strip(c).strip() for c in df.columns]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network traffic prediction (final)")
    parser.add_argument("--csv", help="path to csv", default=DEFAULT_CSV)
    parser.add_argument("--resample", help="resample rule, e.g. '1s' or '1min'", default=None)
    parser.add_argument("--window", help="window size (int)", default=None)
    parser.add_argument("--epochs", help="training epochs", default=None)
    args = parser.parse_args()

    if args.resample:
        RESAMPLE_RULE = args.resample
    if args.window:
        WINDOW_SIZE = int(args.window)
    if args.epochs:
        EPOCHS = int(args.epochs)

    print("Starting. CSV:", args.csv)
    run_pipeline(args.csv)
