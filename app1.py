import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("best.pt")  # Make sure best.pt is in the same folder

st.title("🧏‍♀️ Sign Language Detection App")
st.markdown("Upload an image to detect signs in real-time.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize image for faster detection (optional)
    img_resized = cv2.resize(img, (640, 640))

    # Run prediction with a spinner
    with st.spinner("Detecting signs..."):
        results = model(img_resized)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Convert BGR to RGB for Streamlit
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Detected Sign", use_column_width=True)
