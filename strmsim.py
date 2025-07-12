import streamlit as st
import numpy as np
from PIL import Image
import time
import tensorflow as tf
import pandas as pd
import datetime
import cv2

def is_probably_fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1=30, threshold2=100)
    edge_density = np.sum(edges) / edges.size
    return edge_density > 0.01  # You can tune this value based on test images


# Set up dashboard
st.set_page_config("Fingerprint Dashboard", layout="wide")
st.title("🧬 Fingerprint Prediction Dashboard")

st.markdown("""
> ⚠️ **Disclaimer:**  
> This dashboard is designed for educational and experimental use. Blood group predictions are valid only when uploading clear fingerprint scans. Using unrelated or non-biometric images may produce inaccurate results.  
> Always consult qualified medical professionals for verified blood typing.
""")


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
# uploaded_file = st.file_uploader("📂 Upload Fingerprint Image", type=["bmp", "png", "jpg"])

uploaded_file = st.file_uploader("📂 Upload Fingerprint Image", type=["bmp", "png", "jpg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🔍 Uploaded Fingerprint", width=300)

    # Convert to NumPy array for fingerprint check
    image_np = np.array(img)
    if is_probably_fingerprint(image_np):
        st.success("✅ Image appears valid for fingerprint prediction.")

        with st.spinner("Analyzing scan..."):
            time.sleep(1.5)
            result, confidence = predict_blood_group(img)
            log_prediction(result, confidence, uploaded_file.name)

        if confidence >= 70:
            st.success(f"🎯 **Predicted Blood Group:** `{result}`")
            st.info(f"🔬 Confidence: `{confidence:.2f}%`")
        else:
            st.warning("⚠️ Confidence too low — prediction may be unreliable. Try a clearer scan.")
    else:
        st.warning("⚠️ This image may not be a valid fingerprint. Please upload a biometric scan.")



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
