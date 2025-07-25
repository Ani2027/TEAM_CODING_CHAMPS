# AISOC/Models/Thief_detection/theft.py
from ultralytics import YOLO

try:
    # IMPORTANT: Update with your actual theft model path
    model = YOLO("Models/Thief_dection/theft.pt") 
    model.overrides['conf'] = 0.45
except Exception as e:
    print(f"Error loading theft.pt: {e}")
    model = None

def detect(frame):
    detections = []
    if model is None:
        return detections
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id] # e.g., "Suspicious Activity"
                confidence = float(box.conf[0])
                detections.append({"box": (x1, y1, x2-x1, y2-y1), "label": label, "confidence": confidence})
            except Exception as e:
                print(f"Error processing a theft detection box: {e}")
                continue
    return detections