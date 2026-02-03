import pandas as pd
df =pd.read_csv("/content/consolidated_traffic_data.csv")
df.info()

import numpy as np
df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna()
X=df.drop("traffic_type",axis=1)
y=df["traffic_type"]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_enc=le.fit_transform(y)
label_map=dict(zip(le.classes_,le.transform(le.classes_)))
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# Select only numerical columns for scaling
X_numeric = X.select_dtypes(include=np.number)
X_scaled=scaler.fit_transform(X_numeric)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled , y_enc,test_size=0.2, random_state=42, stratify=y_enc)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train,y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Saved all files.")
import zipfile
import os

# List of files to be zipped (they are in /content/)
files_to_zip = [
    '/content/rf_model.pkl',
    '/content/scaler.pkl',
    '/content/label_encoder.pkl'
]

# Define the path for the output zip file
zip_file_name = 'network_models.zip'
zip_path = os.path.join('/content/', zip_file_name)

# Create the zip file
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file in files_to_zip:
        # Add file to zip, using just the filename in the archive
        zipf.write(file, os.path.basename(file))

print(f"Successfully created zip file: {zip_path}")

import numpy as np

for i in range(10):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)

    pred_class = rf.predict(sample)[0]
    pred_label = le.inverse_transform([pred_class])[0]

    true_label = le.inverse_transform([y_test[idx]])[0]

    print(f"{i+1}. Predicted = {pred_label:15} | Actual = {true_label}")

import pandas as pd
import joblib

selected_features = ['duration','total_fiat','total_biat','min_fiat','min_biat','max_fiat','max_biat','mean_fiat','mean_biat','flowPktsPerSecond','flowBytesPerSecond','min_flowiat','max_flowiat','mean_flowiat','std_flowiat','min_active','mean_active','max_active','std_active','min_idle','mean_idle','max_idle','std_idle']
# assume you have X_test, y_test from train_test_split
probs = rf.predict_proba(X_test)
preds = rf.predict(X_test)
labels = le.inverse_transform(preds)

# create a “real-time like” dataframe
df_sim = pd.DataFrame(X_test, columns=selected_features)
df_sim["Predicted"] = labels

# add a fake timestamp to simulate streaming
import time
base = time.time()
df_sim["Timestamp"] = [base + i for i in range(len(df_sim))]

df_sim.to_csv("simulated_stream.csv", index=False)
print("simulated_stream.csv saved!")