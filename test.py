# inspect_csv.py
import pandas as pd, numpy as np, os, sys

CSV = r"C:\Users\venka\.cache\kagglehub\datasets\ravikumargattu\network-traffic-dataset\versions\2\Midterm_53_group.csv"
print("Checking file:", CSV)
print("Exists:", os.path.exists(CSV))

df = pd.read_csv(CSV, low_memory=False)
df.columns = [c.strip().strip('"') for c in df.columns]
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))
print("\nLast 5 rows:")
print(df.tail(5).to_string(index=False))

# Inspect Time column
if "Time" in df.columns:
    times = pd.to_numeric(df["Time"].astype(str).str.replace('"',''), errors="coerce")
    print("\nTime column stats (numeric coercion):")
    print("  non-null:", times.notna().sum())
    print("  min:", times.min(), "max:", times.max())
    print("  sample values:", times.dropna().head(10).tolist()[:10])
    # show span in seconds
    if times.notna().sum() > 0:
        span = times.max() - times.min()
        print("  span (seconds):", span)
    # quick resample: build datetimes using offsets from min
    base = pd.Timestamp.now().normalize()
    dt_index = base + pd.to_timedelta(times.fillna(0) - times.min(), unit='s')
    s = pd.DataFrame({"len": pd.to_numeric(df.get("Length", 0), errors="coerce").fillna(0).astype(float).values}, index=pd.DatetimeIndex(dt_index))
    res = s.resample("1S").sum()
    print("\nAfter 1-second resample: rows =", len(res), "range:", res.index[0], "->", res.index[-1])
    print("Resample sample (first 10 rows):")
    print(res.head(10).to_string())
else:
    print("No 'Time' column found in CSV.")
