import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("best.pt")  # best.pt should be in the same folder as app.py

def predict(image):
    """
    image: numpy array from webcam
    returns: annotated image, predicted label string
    """
    results = model(image)
    label_text = "No Sign Detected"

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5:  # confidence threshold to filter false positives
                label = model.names[cls]
                label_text = label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label text above the box
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return image, label_text

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(source="webcam", tool="editor", type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Textbox(label="Predicted Sign")],
    live=True
)

iface.launch()
