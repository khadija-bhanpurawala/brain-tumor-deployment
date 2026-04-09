import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="MRI Brain Tumor Detection", page_icon="🧠")

# 2. Load the Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mri_brain_tumor_model.h5')
    return model

model = load_model()

# 3. Class Names (MATCHED WITH YOUR TRAINING)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Optional: Clean display names
DISPLAY_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 4. App UI
st.title("🧠 MRI Brain Tumor Detection")
st.write("Upload an MRI scan to detect the presence and classification of a brain tumor.")

# 5. File Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

    st.write("Analyzing...")

    # 6. Preprocessing (MATCHED WITH TRAINING)
    img = image.resize((224, 224))   # ✅ Correct size
    img_array = np.array(img)

    img_array = img_array / 255.0    # ✅ IMPORTANT normalization
    img_array = np.expand_dims(img_array, axis=0)

    # 7. Prediction
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # 8. Output
        st.success(f"**Prediction:** {DISPLAY_NAMES[predicted_class_index]}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")