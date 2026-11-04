import streamlit as st  # Pehle yahan 'as tf' tha, usay 'as st' kar dein
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page Configuration
st.set_page_config(page_title="Helmet Detector", page_icon="🪖", layout="centered")

st.title("🪖 Helmet Detection App")
st.write("Apni photo upload karein taake model check kar sake ke helmet pehna hua hai ya nahi.")

# 1. Model Load karein
# Apne trained model file ka naam yahan likhein (e.g., 'best.pt' ya 'helmet_model.pt')
MODEL_PATH = "best.pt" 

@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(format(f"Model load karne mein masala hua: {e}"))
        return None

model = load_model(MODEL_PATH)

# 2. File Uploader Component
uploaded_file = st.file_uploader("Koi bhi tasveer (JPG, JPEG, PNG) select karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image ko open aur display karein
    image = Image.open(uploaded_file)
    
    st.subheader("Original Image")
    st.image(image, caption="Aap ki upload ki hui photo", use_container_width=True)
    
    if model is not None:
        with st.spinner("Model tasveer ko scan kar raha hai... Please wait..."):
            # Image ko OpenCV format (numpy array) mein convert karein
            input_image = np.array(image)
            
            # YOLO model se predict karwayein
            results = model(input_image)
            
            # Results ko draw karein (YOLO automatically bounding boxes draw kar deta hai)
            res_plotted = results[0].plot()
            
            # BGR se RGB convert karein kyunki OpenCV BGR use karta hai aur Streamlit RGB
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.success("Scanning mukammal!")
            
            # 3. Output Display karein
            st.subheader("Detection Result")
            st.image(res_rgb, caption="Helmet Detection Output", use_container_width=True)
    else:
        st.warning("Model file sahi se load nahi ho saki. Check karein ke 'best.pt' file mojood hai.")
