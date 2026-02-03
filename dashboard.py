import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scapy.all import rdpcap

# ------------------------ SESSION STATE INIT ------------------------
if "last_category" not in st.session_state:
    st.session_state.last_category = "Unknown"

if "last_update" not in st.session_state:
    st.session_state.last_update = "Not updated"

if "packets" not in st.session_state:
    st.session_state.packets = 0


# ------------------------ THEME / PAGE STYLE ------------------------
st.set_page_config(
    page_title="OPTIC Dashboard",
    layout="wide",
)

st.markdown("""
    <style>
        .title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            color: brown;
        }
        .divider {
            margin-top: 10px;
            margin-bottom: 20px;
            border-top: 2px solid #4F46E5;
        }
        .metric-box {
            background-color: orange;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------ TITLE ------------------------
st.markdown('<div class="title">🔮 OPTIC Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ------------------------ TRAFFIC PREDICTION (STATIC DEMO) ------------------------
st.subheader("📈 Traffic Prediction")

data = pd.DataFrame({
    "time": list(range(10)),
    "bandwidth": np.random.randint(20, 100, size=10)
})

fig = px.line(data, x="time", y="bandwidth", markers=True, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# ------------------------ TRAFFIC CATEGORIZATION (SIMULATED) ------------------------
st.subheader("🧠 Traffic Categorization (Simulated Demo)")

uploaded_csv = st.file_uploader("Upload Traffic Data CSV", type=["csv"], key="categorizer_uploader")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.write("📄 Input Data Preview:")
    st.dataframe(df.head())

    # List of real traffic categories
    all_categories = [
        "BROWSING", "VPN-BROWSING", "VOIP", "VPN-VOIP", "VPN-FT",
        "P2P", "FT", "VPN-P2P", "VPN-CHAT", "CHAT",
        "VPN-MAIL", "MAIL", "STREAMING", "VPN-STREAMING"
    ]

    # Realistic dataset frequency distribution
    traffic_distribution = {
        "BROWSING": 10000,
        "VPN-BROWSING": 10000,
        "VOIP": 6485,
        "VPN-VOIP": 5576,
        "VPN-FT": 4704,
        "P2P": 4000,
        "FT": 3975,
        "VPN-P2P": 3415,
        "VPN-CHAT": 2839,
        "CHAT": 2505,
        "VPN-MAIL": 2444,
        "MAIL": 1364,
        "STREAMING": 1284,
        "VPN-STREAMING": 1115
    }

    # Normalize to probabilities
    total = sum(traffic_distribution.values())
    probabilities = [traffic_distribution[c] / total for c in all_categories]

    # Simulated ML prediction
    df["Predicted Category"] = np.random.choice(all_categories, size=len(df), p=probabilities)

    st.success("Prediction Completed ✔ (Simulated Output)")
    st.dataframe(df)

    # Distribution chart
    fig = px.histogram(
        df,
        x="Predicted Category",
        title="Traffic Category Distribution",
        color="Predicted Category"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Determine most common category
    counts = df["Predicted Category"].value_counts()
    top_class = counts.idxmax()
    st.metric("Most Common Category", top_class)

    # --- UPDATE SYSTEM-WIDE STATUS ---
    st.session_state.last_category = top_class
    st.session_state.last_update = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.packets = len(df)

st.markdown("---")

# ------------------------ SYSTEM STATUS (NOW AT BOTTOM & UPDATING) ------------------------
st.subheader("📊 System Status Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Traffic Category", st.session_state.last_category)

with col2:
    st.metric("Status", "NORMAL")

with col3:
    st.metric("Packets Processed", st.session_state.packets)

st.caption(f"Last Updated: {st.session_state.last_update}")
