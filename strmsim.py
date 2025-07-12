import streamlit as st
import numpy as np
from PIL import Image
import time
import tensorflow as tf
import pandas as pd
import datetime

# Set up dashboard
st.set_page_config("Fingerprint Dashboard", layout="wide")
st.title("🧬 Fingerprint Prediction Dashboard")

# Global variables
class_names = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Load model once with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fingerprint_model.h5")

model = load_model()

# Initialize session state if missing
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# Image preprocessing
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Logging prediction
def log_prediction(result, confidence, image_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_log.append({
        "Timestamp": timestamp,
        "Image Name": image_name,
        "Predicted Blood Group": result,
        "Confidence (%)": round(confidence, 2)
    })

# Blood group prediction
def predict_blood_group(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("📂 Upload Fingerprint Image", type=["bmp", "png", "jpg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🔍 Uploaded Fingerprint", width=300)

    with st.spinner("Analyzing scan..."):
        time.sleep(1.5)
        result, confidence = predict_blood_group(img)
        log_prediction(result, confidence, uploaded_file.name)

    st.success(f"🎯 **Predicted Blood Group:** `{result}`")
    st.info(f"🔬 Confidence: `{confidence:.2f}%`")

# Display log
st.subheader("📊 Prediction Log")
if st.session_state.prediction_log:
    df_log = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df_log)

    # Download CSV
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Save Log to CSV", csv, "prediction_log.csv", "text/csv")

    # Clear log
    if st.button("🗑️ Clear Prediction Log"):
        st.session_state.prediction_log.clear()
        st.warning("Prediction log has been cleared.")
else:
    st.info("No predictions made yet. Upload an image to start.")
