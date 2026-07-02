import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource(show_spinner="Downloading & loading model...")
def load_model():

    if not os.path.exists(MODEL_PATH):
        gdown.download(
            "https://drive.google.com/uc?id=1ABfmnXOOBtJZ38I-jXS-RcVyY4Xsw4DA",
            MODEL_PATH,
            quiet=False
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tabriz Air Pollution Predictor",
    page_icon="🌫️",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "airPollution_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# Order must match the columns used to fit the scaler / model in the notebook
FEATURE_ORDER = [
    "air_temperature",
    "dewpoint",
    "wind_direction_corr",
    "wind_speed",
    "relative_pressure",
    "PM10",
]


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def pm25_category(value: float):
    """Rough EPA-style PM2.5 categories, just for a friendlier readout."""
    if value <= 12:
        return "Good", "#2ecc71"
    if value <= 35.4:
        return "Moderate", "#f1c40f"
    if value <= 55.4:
        return "Unhealthy for Sensitive Groups", "#e67e22"
    if value <= 150.4:
        return "Unhealthy", "#e74c3c"
    if value <= 250.4:
        return "Very Unhealthy", "#8e44ad"
    return "Hazardous", "#7f1d1d"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌫️ Tabriz Air Pollution — PM2.5 Predictor")
st.markdown(
    "Enter the current weather readings below and the model will estimate "
    "the **PM2.5** pollution level (µg/m³)."
)

# ----------------------------------------------------------------------------
# Try loading model/scaler, show a clear message if missing
# ----------------------------------------------------------------------------
model_error = None
scaler_error = None

try:
    model = load_model()
except FileNotFoundError:
    model_error = MODEL_PATH.name
except Exception as e:
    model_error = f"{MODEL_PATH.name} ({e})"

try:
    scaler = load_scaler()
except FileNotFoundError:
    scaler_error = SCALER_PATH.name
except Exception as e:
    scaler_error = f"{SCALER_PATH.name} ({e})"

if model_error or scaler_error:
    missing = [x for x in [model_error, scaler_error] if x]
    st.error(
        "Could not find/load: **"
        + ", ".join(missing)
        + "**.\n\nMake sure these files are in the same folder as `app.py`:\n"
        "- `airPollution_model.pkl`\n"
        "- `scaler.pkl`"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.subheader("Weather Inputs")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        air_temperature = st.slider(
            "Air Temperature (°C)", min_value=-20.0, max_value=45.0, value=15.0, step=0.1
        )
        dewpoint = st.slider(
            "Dewpoint (°C)", min_value=-20.0, max_value=25.0, value=2.0, step=0.1
        )
        wind_direction_corr = st.slider(
            "Wind Direction (°)", min_value=0.0, max_value=360.0, value=100.0, step=1.0
        )

    with col2:
        wind_speed = st.slider(
            "Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=0.5, step=0.05
        )
        relative_pressure = st.slider(
            "Relative Pressure (hPa)", min_value=830.0, max_value=880.0, value=858.0, step=0.1
        )
        pm10 = st.slider(
            "PM10 (µg/m³)", min_value=0.0, max_value=300.0, value=40.0, step=0.5
        )

    submitted = st.form_submit_button("🔍 Predict PM2.5", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [[air_temperature, dewpoint, wind_direction_corr, wind_speed, relative_pressure, pm10]],
        columns=FEATURE_ORDER,
    )

    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    category, color = pm25_category(float(prediction))

    st.markdown("---")
    st.subheader("Result")

    result_col1, result_col2 = st.columns([1, 2])
    with result_col1:
        st.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
    with result_col2:
        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                border-left: 6px solid {color};
                padding: 12px 16px;
                border-radius: 8px;
                font-weight: 600;
                color:{color};
            ">
                {category}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Show input values sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: Random Forest trained on the Tabriz Air Pollution dataset "
    "(air_temperature, dewpoint, wind_direction_corr, wind_speed, relative_pressure, PM10 → PM2.5)."
)