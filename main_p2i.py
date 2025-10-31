import cv2
import time
import math
from ultralytics import YOLO

# Load YOLOv8 model (you can use 'yolov8n.pt' for speed)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Camera resolution
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = frame_w // 2, frame_h // 2

# Smooth laser position variables
laser_x, laser_y = center_x, center_y
smooth_factor = 0.1  # lower = smoother, slower motion

print("✅ YOLO Laser Tracker Simulation Started")

while True:
    success, frame = cap.read()
    if not success:
        print("Camera error")
        break

    results = model(frame, verbose=False)
    detected = False
    target_x, target_y = center_x, center_y

    # Filter detections for humans, vehicles, animals
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls].lower()

        if any(keyword in label for keyword in [
            "person", "car", "bus", "truck", "dog", "cat",
            "bird", "horse", "cow", "sheep"
        ]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2
            detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # track first detected object only

    # Smooth laser motion toward target
    laser_x = int(laser_x + smooth_factor * (target_x - laser_x))
    laser_y = int(laser_y + smooth_factor * (target_y - laser_y))

    # Draw laser dot
    cv2.circle(frame, (laser_x, laser_y), 8, (0, 0, 255), -1)

    # (Pan/Tilt display removed)
    # cv2.putText(frame, f"Pan: {pan_angle:.1f} Tilt: {tilt_angle:.1f}",
    #             (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Laser Tracker Simulation", frame)

    # print(f"Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°", end="\r")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n🟢 Simulation Ended.")
