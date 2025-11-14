
import os
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
# Replaced slow OneClassSVM (rbf) with IsolationForest
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import register_keras_serializable

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Using device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

DATA_DIR = r"C:\Users\jayar\Documents\MAIN_PROJECT\New_Data"
ARTIFACT_DIR = os.path.join(DATA_DIR, "model_artifacts_full")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def safe_read(path):
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path, low_memory=False)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            return pd.DataFrame()
        print(f"Loaded: {Path(path).name} {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️ Failed: {path} ({e})")
        return pd.DataFrame()

# --- Load files (sample large files during read to keep memory reasonable) ---
dfs = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith(".csv") or f.endswith(".parquet"):
        fp = os.path.join(DATA_DIR, f)
        df_tmp = safe_read(fp)
        if len(df_tmp) > 100000:
            df_tmp = df_tmp.sample(100000, random_state=42)
        dfs.append(df_tmp)

if not dfs:
    raise RuntimeError("❌ No files loaded. Check DATA_DIR path and files.")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined shape: {df.shape}")

# --- Label normalization mapping ---
attack_map = {
    "benign": "benign",
    "normal": "benign",
    "background": "benign",
    "ddos": "ddos",
    "dos": "dos",
    "botnet": "botnet",
    "udp": "udp flood",
    "icmp": "icmp flood",
    "xss": "web attack xss",
    "sql": "web attack sql injection",
    "brute": "web attack brute force",
    "syn": "syn flood",
    "slowloris": "slowloris",
    "nmap": "nmapscan",
    "ssh": "ssh brute force",
    "mixed": "mixedscenario",
    "pulsed": "ddos pulsed syn",
    "multiport": "ddos udp multiport",
}

label_candidates = ['label', 'attack', 'class', 'category', 'target']
label_col = next((c for c in df.columns if any(c.lower() == cand for cand in label_candidates)), None)
if label_col is None:
    raise ValueError("❌ No label column found!")

df[label_col] = df[label_col].astype(str).str.strip().str.lower()

def normalize_label(x):
    for k, v in attack_map.items():
        if k in x:
            return v
    return "unknown attack"

df["Normalized_Label"] = df[label_col].apply(normalize_label)
df["__y__"] = df["Normalized_Label"].apply(lambda s: 0 if s == "benign" else 1)

# de-fragment dataframe after inserts
df = df.copy()
print("Label counts:\n", df["Normalized_Label"].value_counts())

# Drop ID-like columns (timestamps, IPs, etc.)
drop_keywords = ['timestamp', 'time', 'src_ip', 'dst_ip', 'source', 'destination', 'mac', 'ipv6', 'flow_id']
drop_cols = [c for c in df.columns if any(k in c.lower() for k in drop_keywords)]
print("Dropping ID-like columns:", drop_cols)
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Build numeric feature matrix
X_df = df.drop(columns=[label_col, "Normalized_Label", "__y__"], errors="ignore")
cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
for c in cat_cols:
    try:
        X_df[c] = X_df[c].astype("category").cat.codes
    except Exception:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.apply(pd.to_numeric, errors="coerce")
print("Numeric columns:", X_df.shape[1])
print("Any NaN before impute:", X_df.isna().values.any())
print("Per-column NaN counts (top 10):")
print(X_df.isna().sum().sort_values(ascending=False).head(10))

# Impute and scale
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_df)
X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

scaler_robust = RobustScaler()
X_rob = scaler_robust.fit_transform(X_imp)

scaler_maxabs = MaxAbsScaler()
X_scaled = scaler_maxabs.fit_transform(X_rob)

# Clip extreme values (optional)
max_clip = 1.0
X_scaled = np.clip(X_scaled, -max_clip, max_clip)

y = df["__y__"].values

# Persist preprocessing artifacts
joblib.dump(imputer, os.path.join(ARTIFACT_DIR, "imputer.joblib"))
joblib.dump(scaler_robust, os.path.join(ARTIFACT_DIR, "scaler_robust.joblib"))
joblib.dump(scaler_maxabs, os.path.join(ARTIFACT_DIR, "scaler_maxabs.joblib"))
print("Saved imputer & scalers")

print("X_scaled stats: min/max/median")
print(np.nanmin(X_scaled), np.nanmax(X_scaled), np.nanmedian(X_scaled))
print("X_scaled percentiles:", np.percentile(X_scaled, [0,1,5,25,50,75,95,99,100]))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Prepare benign-only training set for unsupervised learners
X_train_n = X_train[y_train == 0]
n_benign = len(X_train_n)
if n_benign < 100:
    raise RuntimeError("❌ Not enough benign samples for unsupervised training. Found: {}".format(n_benign))

val_frac = 0.2
val_count = max(1, int(len(X_train_n) * val_frac))
X_val_n = X_train_n[:val_count]
print("Benign-only training rows:", len(X_train_n))
input_dim = X_train.shape[1]
print("Input dim:", input_dim)
print("Any NaN in X_train_n?", np.isnan(X_train_n).any())
print("Any Inf in X_train_n?", np.isinf(X_train_n).any())

X_train_n = np.nan_to_num(X_train_n, nan=0.0, posinf=0.0, neginf=0.0)
X_val_n = np.nan_to_num(X_val_n, nan=0.0, posinf=0.0, neginf=0.0)

# --- VAE setup (unchanged except small safety clips) ---
@register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

enc_in = keras.Input(shape=(input_dim,))
h = layers.Dense(128, activation='relu')(enc_in)
z_mean = layers.Dense(16)(h)
z_log_var = layers.Dense(16)(h)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name="vae_encoder")

dec_in = keras.Input(shape=(16,))
x = layers.Dense(128, activation='relu')(dec_in)
out = layers.Dense(input_dim, activation='tanh')(x)
decoder = keras.Model(dec_in, out, name="vae_decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        recon = self.decoder(z)
        z_log_var_safe = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl = -0.5 * tf.reduce_mean(1 + z_log_var_safe - tf.square(z_mean) - tf.exp(z_log_var_safe))
        self.add_loss(self.kl_weight * kl)
        return recon

vae = VAE(encoder, decoder, kl_weight=1e-3)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

print("Training VAE... this may take a while (but not hours).")
history = vae.fit(X_train_n, X_train_n, epochs=12, batch_size=128, verbose=2)
encoder.save(os.path.join(ARTIFACT_DIR, "vae_encoder.keras"))
decoder.save(os.path.join(ARTIFACT_DIR, "vae_decoder.keras"))

recon = vae.predict(X_val_n)
vae_loss = np.mean(np.square(X_val_n - recon), axis=1)
vae_thresh = np.percentile(vae_loss, 99)
print(f"VAE threshold (percentile=99): {vae_thresh:e}")

# --- LSTM autoencoder (sequence) (unchanged logic, may be skipped if not enough rows) ---
timesteps = 5
if len(X_train_n) <= timesteps:
    print("Not enough benign rows to form LSTM sequences; skipping LSTM training.")
    lstm_thresh = float('nan')
else:
    X_seq = np.array([X_train_n[i:i+timesteps] for i in range(len(X_train_n)-timesteps)])
    X_val_seq = np.array([X_val_n[i:i+timesteps] for i in range(len(X_val_n)-timesteps)]) if len(X_val_n) > timesteps else X_seq[:max(1, len(X_seq)//10)]

    print("LSTM seq shapes:", X_seq.shape, X_val_seq.shape)
    lstm_in = keras.Input(shape=(timesteps, input_dim))
    x = layers.LSTM(64, activation='relu', return_sequences=True)(lstm_in)
    x = layers.LSTM(32, activation='relu', return_sequences=False)(x)
    rep = layers.RepeatVector(timesteps)(x)
    x = layers.LSTM(32, activation='relu', return_sequences=True)(rep)
    x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(input_dim, activation='tanh'))(x)
    lstm_ae = keras.Model(lstm_in, out)
    lstm_ae.compile(optimizer='adam', loss='mse')

    # Use a modest number of epochs; training LSTM can be expensive depending on data size
    lstm_ae.fit(X_seq, X_seq, epochs=10, batch_size=64, validation_data=(X_val_seq, X_val_seq), verbose=2)
    lstm_ae.save(os.path.join(ARTIFACT_DIR, "lstm_autoencoder.keras"))

    X_val_pred = lstm_ae.predict(X_val_seq)
    lstm_loss = np.mean(np.square(X_val_seq - X_val_pred), axis=(1, 2))
    lstm_thresh = np.percentile(lstm_loss, 99)
    print(f"LSTM threshold (percentile=99): {lstm_thresh:e}")

# --- Replacement for slow OneClassSVM (use IsolationForest, much faster and scalable) ---
print("Training IsolationForest (fast SVDD alternative)...")
# Limit the number of benign training rows used for IsolationForest to keep memory/time reasonable
MAX_SVDD_ROWS = 50000  # you can lower this if you have less RAM or want faster runs

svdd_sample = X_train_n
if len(X_train_n) > MAX_SVDD_ROWS:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_n), MAX_SVDD_ROWS, replace=False)
    svdd_sample = X_train_n[idx]

# IsolationForest parameters: n_jobs=-1 uses all cores
svdd = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
svdd.fit(svdd_sample)

# We compute scores on the sample used for training to get a baseline threshold
try:
    # score_samples returns higher for inliers; keep mean as threshold baseline
    svdd_scores = svdd.score_samples(svdd_sample)
    svdd_thresh = float(np.mean(svdd_scores))
except Exception as e:
    print("⚠️ Error computing svdd_scores:", e)
    svdd_thresh = float('nan')

joblib.dump(svdd, os.path.join(ARTIFACT_DIR, "svdd_isolation_forest.pkl"))
print(f"IsolationForest trained and saved. Threshold(mean score): {svdd_thresh}")

# --- Binary LightGBM training (kept reasonable) ---
print("Training binary LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "n_jobs": -1}
# keep num_boost_round moderate to save time; increase if you want higher performance
lgb_binary = lgb.train(params, train_data, num_boost_round=100)
lgb_binary.save_model(os.path.join(ARTIFACT_DIR, "lgb_binary.txt"))
print("Saved binary LightGBM")

# --- Multiclass LightGBM for attack classification ---
labels = df["Normalized_Label"].astype('category')
attack_classes = list(labels.cat.categories)
y_multi = labels.cat.codes.values
num_classes = len(attack_classes)
attack_train = lgb.Dataset(X_scaled, label=y_multi)
attack_params = {"objective": "multiclass", "num_class": num_classes, "learning_rate": 0.05, "num_leaves": 31, "n_jobs": -1}
lgb_multi = lgb.train(attack_params, attack_train, num_boost_round=200)
lgb_multi.save_model(os.path.join(ARTIFACT_DIR, "lgb_attack_multiclass.txt"))
joblib.dump(attack_classes, os.path.join(ARTIFACT_DIR, "attack_classes.joblib"))
print(f"Saved multiclass LightGBM and attack class list: {attack_classes}")

# --- Meta and final save ---
meta = {
    "vae_thresh": float(vae_thresh),
    "lstm_thresh": float(lstm_thresh) if not np.isnan(lstm_thresh) else None,
    "svdd_thresh": float(svdd_thresh),
    "svdd_type": "IsolationForest",
    "svdd_sample_rows": int(len(svdd_sample))
}
json.dump(meta, open(os.path.join(ARTIFACT_DIR, "meta.json"), "w"))
print("Saved meta.json with thresholds")
print("✅ All artifacts saved successfully to:", ARTIFACT_DIR)
