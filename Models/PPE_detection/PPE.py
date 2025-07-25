# AISOC/Models/PPE_detection/PPE.py
from ultralytics import YOLO

try:
    # IMPORTANT: Update with your actual PPE model path
    model = YOLO("Models/PPE_detection/PPE.pt") 
    model.overrides['conf'] = 0.45
except Exception as e:
    print(f"Error loading PPE.pt: {e}")
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
                label = model.names[cls_id] # e.g., "No Helmet", "No Vest"
                confidence = float(box.conf[0])
                detections.append({"box": (x1, y1, x2-x1, y2-y1), "label": label, "confidence": confidence})
            except Exception as e:
                print(f"Error processing a PPE detection box: {e}")
                continue
    return detections