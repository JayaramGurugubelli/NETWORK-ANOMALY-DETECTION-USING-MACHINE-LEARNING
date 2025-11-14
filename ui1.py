import os
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Paths to artifacts
# -----------------------------
ARTIFACTS_PATH = r"C:\Users\jayar\Documents\MAIN_PROJECT\New_Data\model_artifacts_full"

IMPUTER_PATH = os.path.join(ARTIFACTS_PATH, "imputer.joblib")
SCALER_ROBUST_PATH = os.path.join(ARTIFACTS_PATH, "scaler_robust.joblib")
SCALER_MAXABS_PATH = os.path.join(ARTIFACTS_PATH, "scaler_maxabs.joblib")

LGB_BINARY_PATH = os.path.join(ARTIFACTS_PATH, "lgb_binary.txt")
LGB_MULTI_PATH = os.path.join(ARTIFACTS_PATH, "lgb_attack_multiclass.txt")

LSTM_MODEL_PATH = os.path.join(ARTIFACTS_PATH, "lstm_autoencoder.keras")
SVDD_PATH = os.path.join(ARTIFACTS_PATH, "svdd_isolation_forest.pkl")

VAE_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "vae_encoder.keras")
VAE_DECODER_PATH = os.path.join(ARTIFACTS_PATH, "vae_decoder.keras")

ATTACK_CLASSES_PATH = os.path.join(ARTIFACTS_PATH, "attack_classes.joblib")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    artifacts = {}

    def safe_load(name, path, loader):
        try:
            artifacts[name] = loader(path)
        except:
            artifacts[name] = None

    safe_load("imputer", IMPUTER_PATH, joblib.load)
    safe_load("scaler_robust", SCALER_ROBUST_PATH, joblib.load)
    safe_load("scaler_maxabs", SCALER_MAXABS_PATH, joblib.load)

    safe_load("lgb_binary", LGB_BINARY_PATH, joblib.load)
    safe_load("lgb_multi", LGB_MULTI_PATH, joblib.load)

    safe_load("svdd", SVDD_PATH, joblib.load)
    safe_load("attack_classes", ATTACK_CLASSES_PATH, joblib.load)

    try:
        artifacts["lstm"] = load_model(LSTM_MODEL_PATH)
    except:
        artifacts["lstm"] = None

    try:
        artifacts["vae_encoder"] = load_model(VAE_ENCODER_PATH, compile=False)
    except:
        artifacts["vae_encoder"] = None

    try:
        artifacts["vae_decoder"] = load_model(VAE_DECODER_PATH, compile=False)
    except:
        artifacts["vae_decoder"] = None

    return artifacts


# -----------------------------
# Make columns unique
# -----------------------------
def make_columns_unique(df):
    cols = pd.Series(df.columns)
    for dup in df.columns[df.columns.duplicated(keep=False)]:
        dups = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dups):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df


# -----------------------------
# Intelligent Attack Guessing Engine
# -----------------------------
def guess_attack_type(df):
    pkt_rate = df["Flow Pkts/s"].mean() if "Flow Pkts/s" in df else 0
    avg_fwd = df["TotLen Fwd Pkts"].mean() if "TotLen Fwd Pkts" in df else 0
    avg_bwd = df["TotLen Bwd Pkts"].mean() if "TotLen Bwd Pkts" in df else 0
    flags = "".join(df["Flags"].astype(str).unique()).upper() if "Flags" in df else ""

    unique_dports = df["Dst Port"].nunique() if "Dst Port" in df else 0

    # ---- Heuristic rules ----
    if pkt_rate > 5000 or avg_fwd < 200:
        return "Likely DoS/DDoS Flooding (UDP/SYN Flood)"

    if unique_dports > 40:
        return "Likely Port Scan / Reconnaissance"

    if "SYN" in flags and avg_fwd < 150:
        return "Likely SYN Flood or Port Scan"

    if avg_fwd > 50000 or avg_bwd > 50000:
        return "Possible Data Exfiltration"

    if "FIN" in flags or "URG" in flags or "PSH" in flags:
        return "Suspicious TCP Behavior"

    return "Potential Malicious Activity"


# -----------------------------
# Pattern-based anomaly detection
# -----------------------------
def detect_anomalies_no_label(df, sample_rows=10):
    results = {}

    if "Source" in df.columns:
        src_counts = df["Source"].value_counts()
        results["top_sources"] = src_counts.head(10)
        threshold = 0.1 * len(df)
        results["suspected_attackers"] = src_counts[src_counts > threshold]

    if "Destination" in df.columns:
        results["top_destinations"] = df["Destination"].value_counts().head(10)

    if "Protocol" in df.columns:
        results["protocol_distribution"] = df["Protocol"].value_counts()

    sample_df = df.nlargest(sample_rows, "Length") if "Length" in df.columns else df.head(sample_rows)
    sample_df["Predicted_Attack"] = sample_df.apply(heuristic_predict_attack, axis=1)

    results["sample_suspicious"] = make_columns_unique(sample_df)
    return results


# -----------------------------
# Heuristic Attack Prediction (old)
# -----------------------------
def heuristic_predict_attack(row):
    length = row.get("Length", 0)
    proto = str(row.get("Protocol", "")).upper()
    flags = str(row.get("Flags", "")).upper()

    if "ICMP" in proto and length > 1000:
        return "ICMP Flood Attack"
    elif "TCP" in proto and length > 2000:
        return "TCP Flood Attack"
    elif length > 1500:
        return "DDoS Attack"
    elif "SYN" in flags and length < 200:
        return "Port Scan"
    elif any(f in flags for f in ["FIN", "URG", "PSH"]):
        return "Suspicious Traffic"
    else:
        return "Potential Malicious Activity"


# -----------------------------
# UPDATED — Supervised anomaly detection
# -----------------------------
def detect_anomalies_with_label(df, sample_rows=10):

    # find label column
    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    if label_col is None:
        return detect_anomalies_no_label(df)

    labels = df[label_col].astype(str).str.strip().str.upper()

    missing_vals = ["", " ", "NO LABEL", "UNKNOWN", "N/A", "NONE", "NULL"]

    cleaned = []
    for val in labels:
        if val in missing_vals:
            cleaned.append(guess_attack_type(df))  # intelligent guess
        else:
            cleaned.append(val)

    cleaned = pd.Series(cleaned)
    df["Predicted_Attack"] = cleaned.copy()

    counts = cleaned.value_counts()
    attack_types = [x for x in counts.index if x != "BENIGN"]

    if len(attack_types) == 0:
        conclusion = "No attacks detected"
    else:
        conclusion = "Attacks detected: " + ", ".join(attack_types)

    df_anomalies = df[cleaned != "BENIGN"].copy()
    df_anomalies = make_columns_unique(df_anomalies)

    return {
        "df_anomalies": df_anomalies,
        "label_counts": counts,
        "conclusion": conclusion
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hybrid Network Anomaly Detection", layout="wide")
st.title("Hybrid Network Anomaly Detection System")
st.write(
    "Upload a network traffic file (.csv or .parquet). "
    "This app will perform supervised detection (labels) and intelligent attack prediction."
)

uploaded_file = st.file_uploader("📁 Upload CSV/Parquet network traffic file", type=["csv", "parquet"])

if uploaded_file is not None:
    ext = uploaded_file.name.lower().split(".")[-1]
    try:
        df = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write(f"Total Rows: {len(df)}")

    df = make_columns_unique(df)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head(5), use_container_width=True)

    # supervised detection
    if any(col.strip().lower() == "label" for col in df.columns):
        st.subheader("🔍 Supervised Anomaly Detection (Label Column Found)")
        result = detect_anomalies_with_label(df)

        st.write("🚨 Detected Attack Types:")
        st.dataframe(result["label_counts"])

        st.subheader("🧠 Sample Suspicious Packets")
        st.dataframe(result["df_anomalies"].head(10), use_container_width=True)

        st.write("Conclusion:", result["conclusion"])

    else:
        st.subheader("🔍 Pattern-Based Anomaly Detection (No Label Column Found)")
        result = detect_anomalies_no_label(df)

        if "top_sources" in result:
            st.write("🌐 Top Source IPs")
            st.dataframe(result["top_sources"])

        if "suspected_attackers" in result and not result["suspected_attackers"].empty:
            st.write("🚨 Suspected Attackers")
            st.dataframe(result["suspected_attackers"])

        if "protocol_distribution" in result:
            st.write("📡 Protocol Distribution")
            st.dataframe(result["protocol_distribution"])

        st.subheader("⚠️ Sample Suspicious Packets")
        st.dataframe(result["sample_suspicious"])
