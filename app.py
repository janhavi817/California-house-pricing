import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="California House Price Prediction", page_icon="🏡", layout="centered")

# ======== Load model (cached) ========
@st.cache_resource
def load_model():
    return pickle.load(open("xgbmodel.pkl", "rb"))

xgbmodel = load_model()

# ======== App title ========
st.title("🏡 California House Price Predictor")
st.markdown("### Predict median house value based on location and demographics")

# ======== Sidebar inputs ========
st.sidebar.header("Input Features")

# 1️⃣ Median income (original dataset in 10,000s USD → show user-friendly version)
medinc_display = st.sidebar.slider("Median Income (in £1000s)", 0.0, 100.0, 50.0)
MedInc = medinc_display / 10  # Convert back to dataset scale (~8.3 → $83k)

# 2️⃣ House age
HouseAge = st.sidebar.slider("House Age (in years)", 1, 60, 20)

# 3️⃣ Average rooms (no float clutter)
AveRooms = st.sidebar.slider("Average Rooms", 1, 10, 5)

# 4️⃣ Latitude / Longitude
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.0)

# ======== Prediction section ========
if st.button("🔮 Predict House Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, Latitude, Longitude]])

    # Predict
    pred = xgbmodel.predict(input_data)

    # If model was trained on log prices (common for CA dataset)
    # Uncomment below line if outputs seem too small:
    # pred = np.exp(pred)

    predicted_price = float(pred[0]) * 100000  # Convert to USD
    st.success(f"🏠 Estimated House Price: **${predicted_price:,.2f}**")

    st.caption("*(Prices are estimated based on median income and housing features.)*")

# ======== Design tweaks ========
st.markdown("""
<style>
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
</style>
""", unsafe_allow_html=True)
