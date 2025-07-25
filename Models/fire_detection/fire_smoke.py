# AISOC/Models/fire_detection/fire_smoke.py
from ultralytics import YOLO

# Load your model once when the script is imported
try:
    model = YOLO("Models/fire_detection/fire_smoke.pt")
    model.overrides['conf'] = 0.45  # Set a confidence threshold
except Exception as e:
    print(f"Error loading fire_smoke.pt: {e}")
    model = None

def detect(frame):
    """
    Performs fire and smoke detection on a single frame.
    Returns a list of detections.
    """
    detections = []
    if model is None:
        return detections

    # Run inference
    results = model(frame, verbose=False) # Set verbose=False to reduce console output

    # Process results
    for r in results:
        for box in r.boxes:
            try:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Get class and confidence
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    "box": (x1, y1, x2 - x1, y2 - y1), # (x, y, width, height)
                    "label": label,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error processing a detection box: {e}")
                continue
                
    return detections