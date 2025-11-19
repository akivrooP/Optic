# live_dashboard.py
"""
Streamlit live dashboard: actual vs predicted traffic

- Reads `aggregated_stream.csv` (appends by your live_predict_and_store script).
- Loads saved_model/lstm_traffic_model.h5 and saved_model/scaler.save if available.
- Computes one-step-ahead predictions using a sliding window and displays:
    - Live time-series plot: Actual vs Predicted
    - Last prediction and anomaly score
    - Small table of recent alerts (predicted vs actual where relative error > threshold)

Run:
    streamlit run live_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time, math
from datetime import datetime
from tensorflow import keras

# --------- Config (edit if needed) ----------
AGG_CSV = "aggregated_stream.csv"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

WINDOW_SIZE = 30            # must match model training window
PREDICT_AHEAD = 1
REFRESH_SECONDS = 1         # dashboard refresh interval
DISPLAY_LAST_N = 300        # how many recent samples to show in chart
ANOMALY_THRESHOLD = 3.0     # relative error threshold to mark anomaly
# --------------------------------------------

st.set_page_config(page_title="Live Traffic: Actual vs Predicted", layout="wide")
st.title("Live Traffic — Actual vs Predicted")

# sidebar controls
with st.sidebar:
    st.markdown("**Settings**")
    csv_path = st.text_input("Aggregated CSV", AGG_CSV)
    model_path = st.text_input("Model path", MODEL_PATH)
    scaler_path = st.text_input("Scaler path", SCALER_PATH)
    window_size = st.number_input("Window size (samples)", value=WINDOW_SIZE, min_value=1)
    predict_ahead = st.number_input("Predict ahead (steps)", value=PREDICT_AHEAD, min_value=1)
    anomaly_threshold = st.number_input("Anomaly rel-threshold", value=float(ANOMALY_THRESHOLD))
    refresh_seconds = st.number_input("Refresh interval (s)", value=REFRESH_SECONDS, min_value=1)
    display_n = st.number_input("Show last N points", value=DISPLAY_LAST_N, min_value=10)

# helper: load CSV safely
@st.cache_data(ttl=1)
def load_aggregated(csv_file):
    if not os.path.exists(csv_file):
        return None
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None
    # try to find the columns
    col_candidates = [c.lower() for c in df.columns]
    # find time column (iso or epoch)
    if "timestamp_iso" in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp_iso'], utc=True)
    elif "ts_epoch" in df.columns:
        df['ts'] = pd.to_datetime(df['ts_epoch'], unit='s', utc=True)
    elif "ts" in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    else:
        # fallback: use first column as time-like if parseable
        try:
            df['ts'] = pd.to_datetime(df.iloc[:,0], utc=True)
        except Exception:
            # create synthetic index
            df['ts'] = pd.to_datetime(np.arange(len(df)), unit='s', utc=True)
    # find value column (total_bytes or last column)
    if 'total_bytes' in df.columns:
        df['val'] = pd.to_numeric(df['total_bytes'], errors='coerce').fillna(0)
    else:
        df['val'] = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0)
    df = df[['ts','val']].set_index('ts').sort_index()
    return df

# helper: load model+scaler
@st.cache_resource(ttl=60)
def load_model_and_scaler_cached(mpath, spath):
    model = None; scaler = None; msg = ""
    if os.path.exists(spath):
        try:
            scaler = joblib.load(spath)
        except Exception as e:
            msg += f"Failed to load scaler: {e}\n"
    else:
        msg += "Scaler file not found.\n"
    if os.path.exists(mpath):
        try:
            model = keras.models.load_model(mpath, compile=False)
        except Exception as e:
            msg += f"Failed to load model: {e}\n"
    else:
        msg += "Model file not found.\n"
    return model, scaler, msg

# compute rolling one-step predictions
def compute_predictions(series_values, model, scaler, window):
    """
    series_values: 1D numpy array of raw values (original scale)
    returns: preds list aligned to indices >= window (pred for next step)
    """
    if model is None or scaler is None:
        return [None]*len(series_values)
    N = len(series_values)
    preds = [None]*N
    # prepare array for scaling: scaler expects 2D (n,1)
    for i in range(window, N - predict_ahead + 1):
        window_vals = series_values[i-window:i].reshape(-1,1)
        try:
            scaled = scaler.transform(window_vals)
            X = scaled.reshape((1, scaled.shape[0], 1))
            pred_scaled = model.predict(X, verbose=0)
            pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
            preds[i] = pred  # prediction that corresponds to index i (the next value)
        except Exception as e:
            preds[i] = None
    return preds

# main update loop (Streamlit auto-refresh)
st.write("Last update:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))

df = load_aggregated(csv_path)
model, scaler, load_msg = load_model_and_scaler_cached(model_path, scaler_path)

col1, col2 = st.columns([3,1])

with col2:
    st.subheader("Model & data")
    if load_msg:
        st.text(load_msg)
    if model is None or scaler is None:
        st.warning("Model or scaler not loaded. Predictions will be empty until model+scaler are available.")
    if df is None:
        st.error(f"Aggregated CSV not found at: {csv_path}")
        st.stop()
    st.write("Data samples:", len(df))
    st.write("Window size:", window_size)
    st.write("Anomaly threshold:", anomaly_threshold)

with col1:
    st.subheader("Actual vs Predicted (live)")
    # select last display_n rows
    series = df['val'].astype(float)
    if len(series) < 1:
        st.info("No samples yet.")
        st.stop()
    tail = series.tail(display_n)
    vals = tail.values
    # compute preds for the tail window
    preds_full = compute_predictions(series.values, model, scaler, window_size)
    preds_series = pd.Series(preds_full, index=series.index)
    preds_tail = preds_series.tail(display_n)

    plot_df = pd.DataFrame({
        "actual": tail,
        "predicted": preds_tail
    })
    st.line_chart(plot_df)

    # Show last predicted vs actual
    last_time = series.index[-1]
    last_actual = series.iloc[-1]
    last_pred = preds_series.iloc[-1]  # probably None (prediction corresponds to index where window exists)
    # find most recent non-None predicted index
    non_null_preds = preds_series.dropna()
    if len(non_null_preds) > 0:
        pred_idx = non_null_preds.index[-1]
        pred_val = non_null_preds.iloc[-1]
        actual_for_pred = series.loc[pred_idx] if pred_idx in series.index else np.nan
        rel_err = abs(actual_for_pred - pred_val) / (pred_val + 1e-9) if not np.isnan(actual_for_pred) else None
        st.write(f"Most recent prediction for index {pred_idx} → predicted={pred_val:.2f}, actual={actual_for_pred:.2f}, rel_err={rel_err:.2f}")
    else:
        st.info("Not enough history yet to show predictions (need window-size samples).")

    # show a small table of recent anomalies
    st.subheader("Recent anomalies (rel_err > threshold)")
    anomalies = []
    # compute rel errors for all indices where we have preds
    for t, p in non_null_preds.items():
        if p is None:
            continue
        if t in series.index:
            a = series.loc[t]
            rel = abs(a - p) / (p + 1e-9)
            if rel > anomaly_threshold:
                anomalies.append((t, float(a), float(p), float(rel)))
    if anomalies:
        anom_df = pd.DataFrame(anomalies, columns=["timestamp","actual","predicted","rel_err"]).sort_values("timestamp", ascending=False).head(20)
        st.dataframe(anom_df)
    else:
        st.write("No recent anomalies detected.")

# auto-refresh
st.experimental_rerun() if st.button("Refresh Now") else None
st_autorefresh = st.experimental_get_query_params().get("autorefresh", None)
# naive loop: use sleep and rerun via st.experimental_rerun; but Streamlit has st.experimental_singleton caching.
# Use Streamlit's built-in rerun mechanism via a short delay:
time.sleep(refresh_seconds := int(refresh_seconds) if isinstance((refresh_seconds := refresh_seconds), int) else REFRESH_SECONDS)
st.rerun()
