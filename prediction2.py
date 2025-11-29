"""
prediction.py

Robust network traffic prediction. Fixed handling for Wireshark-style CSVs where
"Time" is a seconds-offset float (e.g., 0.000000, 0.784682, ...).

Copy this file over your existing one and run:
python prediction.py --csv <path-to-your-csv>
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
DATA_CSV = "C:\\Users\\venka\\.cache\\kagglehub\\datasets\\ravikumargattu\\network-traffic-dataset\\versions\\2\\Midterm_53_group.csv"     # expected columns: timestamp, total_bytes (or total_traffic)
TIMESTAMP_COL_PREFERRED = "Time"   # the CSV column that holds seconds-offset (your file uses "Time")
VALUE_COL_PREFERRED = "Length"     # packet length column in your CSV
RESAMPLE_RULE = "1S"          # resample to 1-second bins (change to "1Min" for min-level)
WINDOW_SIZE = 30              # history window (in resample units, e.g., seconds)
PREDICT_AHEAD = 1             # predict next 1 interval
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
RUN_STREAMLIT = False   # set True to run streamlit part (use `streamlit run prediction.py`)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# Helpers
# -------------------------
def debug(msg):
    print(msg)

def safe_strip_quotes(s):
    if isinstance(s, str):
        return s.strip().strip('"').strip("'")
    return s

# -------------------------
# Loading & timestamp handling (FIXED)
# -------------------------
def load_csv_as_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    # clean column names
    df.columns = [safe_strip_quotes(c).strip() for c in df.columns]
    return df

def detect_and_prepare(df):
    """
    Prepare DataFrame with DatetimeIndex and single value column.
    Special handling: if TIMESTAMP_COL_PREFERRED exists, treat it as seconds-offset floats
    and convert to datetimes by adding to a base timestamp.
    """
    df = df.copy()
    debug("CSV columns: " + ", ".join(df.columns.tolist()))

    # sanitize string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).apply(safe_strip_quotes)

    # If preferred timestamp column exists, treat as seconds offsets (critical fix)
    dt_index = None
    ts_col_used = None
    if TIMESTAMP_COL_PREFERRED in df.columns:
        debug(f"Found preferred timestamp column '{TIMESTAMP_COL_PREFERRED}'. Treating it as seconds-offset (float).")
        # convert to numeric (floats), coerce invalids to NaN
        time_numeric = pd.to_numeric(df[TIMESTAMP_COL_PREFERRED].astype(str).str.replace('"',''), errors='coerce')
        if time_numeric.isna().all():
            debug(f"'{TIMESTAMP_COL_PREFERRED}' exists but contains no numeric values after coercion.")
        else:
            # build datetime index: choose a base date (use today normalized)
            base = pd.Timestamp.now().normalize()
            # create datetimes by adding offsets in seconds relative to the minimum offset
            offset_seconds = time_numeric.fillna(0) - time_numeric.min()
            dt_index = base + pd.to_timedelta(offset_seconds, unit='s')
            ts_col_used = TIMESTAMP_COL_PREFERRED
            debug(f"Converted '{TIMESTAMP_COL_PREFERRED}' numeric offsets to datetimes using base {base}.")

    # If not created, attempt auto-detection for timestamp-like column names (rare for your dataset)
    if dt_index is None:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("time","date","timestamp","ts"))]
        debug("Timestamp candidates by name: " + ", ".join(candidates))
        for c in candidates:
            # Try parsing as ISO / epoch (coerce)
            parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            if parsed.notna().sum() >= 2:
                dt_index = parsed
                ts_col_used = c
                debug(f"Auto-detected timestamp column '{c}' by parsing.")
                break

    # fallback: build synthetic increasing index (1s spacing)
    if dt_index is None:
        debug("No parsable timestamp column found. Falling back to synthetic increasing DatetimeIndex (1s spacing).")
        n = len(df)
        dt_index = pd.date_range(start=pd.Timestamp.now().normalize(), periods=n, freq='1S')

    # select value column (prefer LENGTH)
    if VALUE_COL_PREFERRED in df.columns:
        val_col = VALUE_COL_PREFERRED
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != ts_col_used]
        if numeric_cols:
            val_col = numeric_cols[0]
            debug(f"Value column '{VALUE_COL_PREFERRED}' not found; using '{val_col}' instead.")
        else:
            # attempt numeric coercion candidates
            potential = []
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace('"',''), errors='raise')
                    potential.append(c)
                except Exception:
                    pass
            potential = [c for c in potential if c != ts_col_used]
            if potential:
                val_col = potential[0]
                debug(f"Using coerced numeric column '{val_col}' as value column.")
            else:
                raise ValueError("No numeric column found to use as traffic value.")

    # Build final DataFrame
    df_out = pd.DataFrame(index=pd.DatetimeIndex(dt_index))
    df_out.index.name = "datetime"
    # coerce value column to numeric
    df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace('"',''), errors='coerce').fillna(0).astype(float)
    df_out[VALUE_COL_PREFERRED] = df[val_col].values
    debug(f"Prepared data: rows={len(df_out)}, range={df_out.index[0]} -> {df_out.index[-1]}")
    return df_out

# -------------------------
# Resample / scale / windows
# -------------------------
def resample_and_fill(df, rule=RESAMPLE_RULE):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for resampling.")
    if len(df) == 0:
        raise ValueError("Empty DataFrame.")
    if rule is None:
        s = df.iloc[:,0].ffill().bfill()
        return s.to_frame(name=df.columns[0])
    # resample by summing packet lengths into bins
    try:
        s = df.iloc[:,0].resample(rule).sum().fillna(0)
        debug(f"Resampled to rule {rule}: {len(s)} rows (index {s.index[0]} -> {s.index[-1]})")
        return s.to_frame(name=df.columns[0])
    except Exception as e:
        debug(f"Resample failed ({e}); returning original series filled.")
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

def create_windows(arr, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD):
    X, y = [], []
    N = len(arr)
    for i in range(N - window_size - predict_ahead + 1):
        X.append(arr[i:i+window_size])
        y.append(arr[i+window_size:i+window_size+predict_ahead])
    X = np.array(X)
    y = np.array(y)
    if y.size == 0:
        return X, y
    if y.shape[-1] == 1:
        y = y.reshape((y.shape[0], 1))
    return X, y

# -------------------------
# Model
# ---------------------
