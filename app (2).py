"""
Streamlit app:  streamlit run app.py
Zaroori files (same folder me): skin_cancer_cnn.keras, class_names.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("skin_cancer_cnn.keras")
CLASS_NAMES_PATH = Path("class_names.json")
IMG_SIZE = (128, 128)  # train.py wala size

FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (mole)",
    "vasc": "Vascular lesion",
}

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


st.title("🩺 Skin Cancer Detection (CNN)")
st.caption("Skin lesion ki image upload karein aur model prediction dekhein.")

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error(
        "Model files nahi mili. Pehle `python train.py` chalayein taake "
        "`skin_cancer_cnn.keras` aur `class_names.json` ban jayein."
    )
    st.stop()

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Image upload karein", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Ye image open nahi ho saki. Doosri JPG/PNG try karein.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    arr = np.array(image.resize(IMG_SIZE), dtype="float32")  # raw 0-255
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(probs))
    code = class_names[top]

    st.subheader(f"Prediction: {FULL_NAMES.get(code, code)}")
    st.write(f"Confidence: **{probs[top] * 100:.2f}%**")

    df = pd.DataFrame(
        {"Probability (%)": probs * 100},
        index=[FULL_NAMES.get(c, c) for c in class_names],
    )
    st.bar_chart(df)

st.warning(
    "⚠️ Ye sirf educational project hai, medical diagnosis nahi. "
    "Kisi bhi shak ki surat me dermatologist se milen."
)
