
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Helmet Detection AI",
    page_icon="⛑️",
    layout="wide"
)

st.title("⛑️ Helmet Detection AI")
st.write("AI-powered Motorcycle Helmet Detection System")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.success("Helmet Detection Model Loaded Successfully!")

option = st.selectbox(
    "Choose Detection Type",
    ["Image Detection", "Video Detection"]
)

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("Detect Helmet", type="primary"):
            with st.spinner("Analyzing image..."):
                results = model.predict(
                    image,
                    conf=0.25,
                    imgsz=640
                )

                result_image = results[0].plot()

                with col2:
                    st.subheader("Detection Result")
                    st.image(
                        result_image,
                        channels="BGR",
                        use_container_width=True
                    )

                if len(results[0].boxes) == 0:
                    st.warning("No helmet-related objects detected.")
                else:
                    st.subheader("Detection Details")

                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = model.names[class_id]

                        st.write(
                            f"**{label}** — "
                            f"Confidence: {confidence:.1%}"
                        )

elif option == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Start Video Detection", type="primary"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(uploaded_video.getvalue())
                input_path = temp_file.name

            try:
                with st.spinner("Processing video..."):

                    results = model.predict(
                        source=input_path,
                        conf=0.25,
                        imgsz=640,
                        save=True,
                        project="runs",
                        name="helmet_detection",
                        exist_ok=True,
                        verbose=False
                    )

                    output_path = os.path.join(
                        "runs",
                        "helmet_detection"
                    )

                    video_files = [
                        f for f in os.listdir(output_path)
                        if f.lower().endswith(
                            (".mp4", ".avi", ".mov", ".mkv")
                        )
                    ]

                    if video_files:
                        result_video = os.path.join(
                            output_path,
                            video_files[0]
                        )

                        st.success("Video detection completed!")
                        st.subheader("Processed Video")

                        st.video(result_video)

                    else:
                        st.error("Processed video was not found.")

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)

st.divider()

st.caption(
    "Educational AI project. Predictions may be incorrect "
    "and should not be treated as definitive safety verification."
)
