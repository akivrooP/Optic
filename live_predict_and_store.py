#!/usr/bin/env python3
"""
live_predict_and_store.py

- Spawns tshark to stream "frame.time_epoch,frame.len"
- Aggregates packets into bins (default 1s)
- Appends aggregated bins to aggregated_stream.csv
- Loads saved LSTM+scaler to predict next bin; compares when actual arrives -> alert
- Optionally retrains periodically from aggregated_stream.csv

Usage:
    cmd.exe> python live_predict_and_store.py --iface 2
    # or pass --csv to use an existing CSV file instead of live capture

Requirements:
    pip install numpy pandas joblib tensorflow scikit-learn
    tshark installed and on PATH
"""

import argparse
import subprocess
import sys
import time
import math
import csv
import os
from datetime import datetime, timezone
import collections
import threading

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# ----------------- Configuration -----------------
AGG_CSV = "aggregated_stream.csv"   # CSV where each 1s/min aggregate is appended
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

DEFAULT_IFACE = None   # must set via --iface argument or use --csv to read file
RESAMPLE_RULE = "1s"   # "1s" or "1m"
WINDOW_SIZE = 30       # history length (in resampled units)
PREDICT_AHEAD = 1
ANOMALY_THRESHOLD = 3.0   # simple rule: flag if |actual - pred| / (pred + eps) > threshold
RETRAIN_AFTER_SECONDS = 0  # if >0, automatically retrain model every N seconds
RETRAIN_AFTER_SAMPLES = 0  # if >0, retrain after this many new samples appended
BATCH_EPOCHS = 10      # epochs when retraining in this script (small by default)
TEST_RATIO = 0.2
# -------------------------------------------------

# thread-safe state
state = {
    "current_bin_start": None,   # int second (or minute index depending on RESAMPLE_RULE)
    "current_bin_total": 0.0,
    "window": collections.deque(maxlen=WINDOW_SIZE),
    "pending_prediction": None,  # (pred_value, target_ts)
    "model": None,
    "scaler": None,
    "appended_since_retrain": 0,
    "last_retrain_time": 0
}

# ---------------- Helpers ----------------
def debug(msg):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    print(f"[{ts}] {msg}")

def load_model_and_scaler():
    """
    Robust loader: loads Keras model with compile=False (avoids deserializing old metrics)
    and loads scaler via joblib. If loading fails, logs error and leaves model/scaler None.
    """
    state["model"] = None
    state["scaler"] = None

    # Load scaler first (joblib)
    try:
        if os.path.exists(SCALER_PATH):
            state["scaler"] = joblib.load(SCALER_PATH)
            debug("Loaded scaler from " + SCALER_PATH)
        else:
            debug("Scaler file not found: " + SCALER_PATH)
    except Exception as e:
        debug(f"Failed to load scaler ({SCALER_PATH}): {e}")
        state["scaler"] = None

    # Load model for inference only (compile=False)
    try:
        if os.path.exists(MODEL_PATH):
            state["model"] = keras.models.load_model(MODEL_PATH, compile=False)
            debug("Loaded Keras model (compile=False) from " + MODEL_PATH)
        else:
            debug("Model file not found: " + MODEL_PATH)
    except Exception as e:
        debug(f"Failed to load model ({MODEL_PATH}): {e}")
        state["model"] = None

    if state["model"] is None or state["scaler"] is None:
        debug("Model or scaler not available — the script will still capture and append aggregates but won't predict until model+scaler are loaded.")


def save_model_and_scaler(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    debug(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

def append_aggregate_csv(ts_epoch, total_bytes):
    # append a line: timestamp_iso, unix_epoch, total_bytes
    iso = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()
    exists = os.path.exists(AGG_CSV)
    with open(AGG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp_iso", "ts_epoch", "total_bytes"])
        writer.writerow([iso, f"{ts_epoch:.6f}", f"{total_bytes:.6f}"])
    state["appended_since_retrain"] += 1

def predict_from_window():
    # returns predicted value for next bin (original scale) or None
    model = state["model"]
    scaler = state["scaler"]
    if model is None or scaler is None:
        return None
    if len(state["window"]) < WINDOW_SIZE:
        return None
    arr = np.array(state["window"], dtype=float).reshape(-1,1)
    scaled = scaler.transform(arr)
    X = scaled.reshape((1, scaled.shape[0], 1))
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
    return pred

# ---------------- Aggregation / prediction loop ----------------
def handle_finished_bin(bin_ts, bin_total):
    """
    bin_ts: unix epoch integer of the bin (e.g., second)
    bin_total: total bytes in this bin
    """
    # 1) append to CSV
    append_aggregate_csv(bin_ts, bin_total)

    # 2) add to sliding window
    state["window"].append(bin_total)

    # 3) If we had a pending prediction for this bin, compare & alert
    if state["pending_prediction"] is not None:
        pred_val, target_ts = state["pending_prediction"]
        if target_ts == bin_ts:
            # compare
            actual = bin_total
            rel = abs(actual - pred_val) / (pred_val + 1e-9)
            if rel > ANOMALY_THRESHOLD:
                debug(f"[ALERT] target_ts={datetime.fromtimestamp(bin_ts)} actual={actual:.2f} pred={pred_val:.2f} rel={rel:.2f}")
            else:
                debug(f"[OK] target_ts={datetime.fromtimestamp(bin_ts)} actual={actual:.2f} pred={pred_val:.2f} rel={rel:.2f}")
            state["pending_prediction"] = None

    # 4) If we have enough window, make prediction for next bin and store pending
    pred = predict_from_window()
    if pred is not None:
        next_ts = bin_ts + (1 if RESAMPLE_RULE.endswith("s") else 60)
        state["pending_prediction"] = (pred, next_ts)
        debug(f"[PRED] for {datetime.fromtimestamp(next_ts)} -> {pred:.2f}")

    # 5) retraining trigger checks
    maybe_trigger_retrain()

def maybe_trigger_retrain():
    # retrain if configured and conditions met
    now = time.time()
    if RETRAIN_AFTER_SECONDS > 0 and now - state["last_retrain_time"] > RETRAIN_AFTER_SECONDS and state["appended_since_retrain"] > 0:
        debug("Retrain timer triggered.")
        # run retrain in a separate thread to avoid blocking capture
        t = threading.Thread(target=retrain_from_csv_and_save, args=(AGG_CSV,))
        t.daemon = True
        t.start()
        state["last_retrain_time"] = now
        state["appended_since_retrain"] = 0
    elif RETRAIN_AFTER_SAMPLES > 0 and state["appended_since_retrain"] >= RETRAIN_AFTER_SAMPLES:
        debug("Retrain sample-count triggered.")
        t = threading.Thread(target=retrain_from_csv_and_save, args=(AGG_CSV,))
        t.daemon = True
        t.start()
        state["last_retrain_time"] = now
        state["appended_since_retrain"] = 0

# ---------------- Retrain function (simple) ----------------
def retrain_from_csv_and_save(csv_path):
    """
    Simple retraining: loads aggregated CSV, trains an LSTM from scratch with small epochs,
    saves model + scaler. This is intentionally lightweight; for production you may want
    more controlled retraining.
    """
    debug("Retrain started: loading CSV...")
    if not os.path.exists(csv_path):
        debug("No aggregated CSV found; abort retrain.")
        return
    df = pd.read_csv(csv_path)
    if "total_bytes" not in df.columns:
        # accept older header names
        df = df.rename(columns={df.columns[-1]: "total_bytes"})
    series = df["total_bytes"].astype(float).values
    if len(series) < WINDOW_SIZE + PREDICT_AHEAD + 2:
        debug("Not enough data to retrain. Need more samples.")
        return
    # prepare windows & scaler
    data = series.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)
    X, y = [], []
    N = len(scaled)
    for i in range(N - WINDOW_SIZE - PREDICT_AHEAD + 1):
        X.append(scaled[i:i+WINDOW_SIZE])
        y.append(scaled[i+WINDOW_SIZE:i+WINDOW_SIZE+PREDICT_AHEAD])
    X = np.array(X)
    y = np.array(y)
    if y.shape[-1] == 1:
        y = y.reshape((y.shape[0], 1))
    # time split
    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # build model
    tf_model = keras.models.Sequential([
        keras.layers.Input(shape=(WINDOW_SIZE,1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(PREDICT_AHEAD)
    ])
    tf_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    debug(f"Retrain: training on {len(X_train)} samples, validating on {len(X_test)}")
    tf_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=BATCH_EPOCHS, batch_size=32, verbose=2)
    # evaluate
    preds_scaled = tf_model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1))
    ytrue = scaler.inverse_transform(y_test.reshape(-1,1))
    rmse = math.sqrt(((preds - ytrue)**2).mean())
    debug(f"Retrain finished: RMSE={rmse:.3f}")
    # save
    save_model_and_scaler(tf_model, scaler)
    # load into runtime
    state["model"] = tf_model
    state["scaler"] = scaler
    debug("Retrain complete and new model loaded into runtime.")

# ---------------- Main capture loop (spawns tshark) ----------------
def spawn_tshark_and_process(iface):
    # build tshark command
    # interface can be index (int) or name string
    cmd = ["tshark", "-i", str(iface), "-T", "fields", "-e", "frame.time_epoch", "-e", "frame.len", "-E", "separator=,", "-l"]
    debug("Spawning tshark: " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    try:
        for raw in p.stdout:
            raw = raw.strip()
            if not raw:
                continue
            # parse lines like: 167xxx.xxxxxx,74
            parts = raw.split(",",1)
            if len(parts) != 2:
                continue
            try:
                ts = float(parts[0])
                length = float(parts[1])
            except:
                continue
            process_packet(ts, length)
    except KeyboardInterrupt:
        debug("KeyboardInterrupt received. Stopping tshark process.")
        p.terminate()
        p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait()

def process_packet(ts_float, length):
    """
    Aggregates packet into current bin depending on RESAMPLE_RULE.
    Supports seconds ('s') or minutes ('m') rules.
    """
    # determine bin width in seconds
    if RESAMPLE_RULE.endswith("s"):
        bin_width = 1
    elif RESAMPLE_RULE.endswith("m"):
        bin_width = 60
    else:
        bin_width = 1

    bin_ts = int(math.floor(ts_float / bin_width) * bin_width)

    if state["current_bin_start"] is None:
        state["current_bin_start"] = bin_ts
        state["current_bin_total"] = 0.0

    if bin_ts == state["current_bin_start"]:
        state["current_bin_total"] += length
    else:
        # previous bin ended
        finished_ts = state["current_bin_start"]
        finished_total = state["current_bin_total"]
        # handle finished bin
        handle_finished_bin(finished_ts, finished_total)
        # start new bin
        state["current_bin_start"] = bin_ts
        state["current_bin_total"] = length

# ---------------- CLI and startup ----------------
def main():
    parser = argparse.ArgumentParser(description="Live tshark -> aggregator -> predictor -> storage")
    parser.add_argument("--iface", help="tshark interface number or name (use tshark -D to list)", default=None)
    parser.add_argument("--csv", help="Use existing CSV instead of spawning tshark (path)", default=None)
    parser.add_argument("--resample", help="Resample rule: '1s' or '1m'", default=None)
    parser.add_argument("--window", help="Window size (int)", default=None)
    parser.add_argument("--retrain_seconds", help="Retrain every N seconds (0 disable)", default=None)
    parser.add_argument("--retrain_samples", help="Retrain after this many appended samples (0 disable)", default=None)
    args = parser.parse_args()

    global RESAMPLE_RULE, WINDOW_SIZE, RETRAIN_AFTER_SECONDS, RETRAIN_AFTER_SAMPLES
    if args.resample:
        RESAMPLE_RULE = args.resample
    if args.window:
        WINDOW_SIZE = int(args.window)
        state["window"] = collections.deque(maxlen=WINDOW_SIZE)
    if args.retrain_seconds:
        RETRAIN_AFTER_SECONDS = int(args.retrain_seconds)
    if args.retrain_samples:
        RETRAIN_AFTER_SAMPLES = int(args.retrain_samples)

    load_model_and_scaler()  # attempt to load existing model/scaler

    if args.csv:
        # If user wants to stream from existing CSV (replay mode)
        debug(f"Replaying CSV {args.csv}")
        replay_csv(args.csv)
    else:
        if args.iface is None:
            debug("Error: no interface specified. Use --iface N (see `tshark -D`).")
            sys.exit(1)
        spawn_tshark_and_process(args.iface)

def replay_csv(csv_path):
    """
    Useful for testing: read a CSV with columns ts_epoch,total_bytes (or similar),
    and feed into the pipeline at real-time speed or fast mode.
    """
    if not os.path.exists(csv_path):
        debug("CSV not found: " + csv_path)
        return
    df = pd.read_csv(csv_path)
    # try to discover column names
    if "ts_epoch" in df.columns:
        tscol = "ts_epoch"
    elif "timestamp" in df.columns:
        tscol = "timestamp"
    elif "Time" in df.columns:
        tscol = "Time"
    else:
        tscol = df.columns[0]
    if "total_bytes" in df.columns:
        valcol = "total_bytes"
    else:
        valcol = df.columns[-1]
    last_ts = None
    for _, row in df.iterrows():
        ts = float(row[tscol])
        val = float(row[valcol])
        # simulate real time
        if last_ts is not None:
            wait = ts - last_ts
            if wait > 0:
                time.sleep(min(wait, 1.0))  # don't wait too long during replay
        process_packet(ts, val)
        last_ts = ts

if __name__ == "__main__":
    main()
