import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model (using pretrained 'yolov8n.pt')
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

# Laser dot parameters
dot_pos = np.array([320, 240], dtype=float)  # start roughly center
smooth_factor = 0.15  # how fast laser moves (lower = smoother)

# Target classes (YOLO names)
target_classes = ["person", "car", "motorbike", "bus", "truck", "dog", "cat", "bird", "cow", "horse"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    target_center = None

    for box in detections:
        cls = int(box.cls[0])
        name = model.names[cls]
        if name in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            target_center = np.array([cx, cy])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # only track one at a time

    # Smooth laser movement
    if target_center is not None:
        dot_pos = (1 - smooth_factor) * dot_pos + smooth_factor * target_center

    # Draw laser dot
    cv2.circle(frame, tuple(dot_pos.astype(int)), 6, (0, 0, 255), -1)

    cv2.imshow("AI Laser Tracker - Phase 1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
