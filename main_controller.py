# AISOC/main_controller.py
import cv2
import os
import json
import datetime
import time
import sys
from twilio.rest import Client
import config

# --- NEW: SendGrid Library ---
import sendgrid
from sendgrid.helpers.mail import Mail

# --- CONFIGURATION ---
sys.path.append('Models/fire_detection')
# ... (rest of your sys.path append lines) ...

from Models.fire_detection.fire_smoke import detect as detect_fire
from Models.PPE_detection.PPE import detect as detect_ppe
from Models.Thief_detection.theft import detect as detect_theft

# --- Twilio Credentials ---
TWILIO_ACCOUNT_SID = config.TWILIO_ACCOUNT_SID  
TWILIO_AUTH_TOKEN = config.TWILIO_AUTH_TOKEN                 
TWILIO_WHATSAPP_FROM = config.TWILIO_WHATSAPP_FROM     
YOUR_WHATSAPP_TO = config.YOUR_WHATSAPP_TO             

# --- NEW: SendGrid Credentials ---
SENDGRID_API_KEY = config.SENDGRID_API_KEY
SENDER_EMAIL = config.SENDER_EMAIL  
RECIPIENT_EMAIL = config.RECIPIENT_EMAIL

# ... (Rest of your configuration is the same) ...
CAMERA_SOURCES = { "CAM-01": "videos/fire1.mp4", "CAM-02": "videos/ppe1.mp4", "CAM-03": "videos/theft1.mp4", "CAM-04": "videos/fire2.mp4" }
OUTPUT_DIR = "output_frames"
LOG_FILE = "logs.json"
PROCESSING_INTERVAL_SECONDS = 0.5
LOG_WRITE_INTERVAL_SECONDS = 5.0
FIRE_ALERT_COOLDOWN_MINUTES = 15
FIRE_ALERT_COOLDOWN_SECONDS = FIRE_ALERT_COOLDOWN_MINUTES * 60

fire_alert_timestamps = {}
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_buffer = []
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- UPDATED Notification Functions ---

def send_whatsapp_alert(message):
    try:
        print(f"📲 Sending real WhatsApp alert...")
        message = twilio_client.messages.create(body=message, from_=TWILIO_WHATSAPP_FROM, to=YOUR_WHATSAPP_TO)
        print(f"✅ WhatsApp alert sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"❌ ERROR: Failed to send WhatsApp alert. Error: {e}")

def send_email_alert(subject, body):
    """Sends a real email alert using the SendGrid API."""
    print(f"📧 Sending real email alert via SendGrid...")
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=RECIPIENT_EMAIL,
        subject=subject,
        html_content=f"<p>{body.replace(chr(10), '<br>')}</p>") # Basic HTML formatting for newlines
    try:
        sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        if response.status_code == 202:
            print(f"✅ Email alert sent successfully!")
        else:
            print(f"❌ ERROR: Failed to send email. Status: {response.status_code} Body: {response.body}")
    except Exception as e:
        print(f"❌ ERROR: Failed to send email alert. Error: {e}")


# --- Alert Handling & Main Application (No changes below this line) ---
# ... (The rest of your main_controller.py is exactly the same) ...
def handle_fire_alert(camera_id, timestamp):
    current_time = timestamp.timestamp()
    last_alert_time = fire_alert_timestamps.get(camera_id, 0)
    if current_time - last_alert_time > FIRE_ALERT_COOLDOWN_SECONDS:
        print(f"🔥 Fire alert condition met for {camera_id}. Sending notifications...")
        fire_alert_timestamps[camera_id] = current_time
        subject = f"URGENT: Fire Detected on {camera_id}"
        message = f"""🚨 *FIRE ALERT!* 🚨

A fire or smoke has been detected on camera *{camera_id}* at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}.

Please review the live feed immediately."""
        send_whatsapp_alert(message)
        send_email_alert(subject, message)

def write_logs_from_buffer():
    global log_buffer
    if not log_buffer: return
    try:
        try:
            with open(LOG_FILE, 'r') as f: existing_logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): existing_logs = []
        updated_logs = log_buffer + existing_logs
        with open(LOG_FILE, 'w') as f: json.dump(updated_logs, f, indent=4)
        log_buffer = []
    except Exception as e: print(f"Error writing logs to file: {e}")

def main():
    print("🚀 Central surveillance controller started. Press Ctrl+C to exit.")
    caps = {cam_id: cv2.VideoCapture(path) for cam_id, path in CAMERA_SOURCES.items()}
    last_log_write_time = time.time()
    detection_functions = {"Fire": detect_fire, "PPE": detect_ppe, "Theft": detect_theft}
    detection_colors = {"Fire": (0, 0, 255), "PPE": (0, 255, 255), "Theft": (255, 0, 255)}
    try:
        while True:
            start_time = time.time()
            for camera_id, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret: continue
                timestamp = datetime.datetime.now()
                for detection_type, detect_func in detection_functions.items():
                    detections = detect_func(frame)
                    if not detections: continue
                    for detection in detections:
                        if detection_type == "Fire":
                            handle_fire_alert(camera_id, timestamp)
                        print(f"🚨 Logged: {detection['label']} detected on {camera_id}")
                        log_buffer.append({ "timestamp": timestamp.isoformat(), "camera_id": camera_id, "detection_type": detection_type, "event_details": detection['label'], "confidence": f"{detection['confidence']:.2f}" })
                        x, y, w, h = detection['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), detection_colors[detection_type], 2)
                        cv2.putText(frame, f"{detection['label']} ({detection['confidence']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_colors[detection_type], 2)
                output_path = os.path.join(OUTPUT_DIR, f"{camera_id.lower()}.jpg")
                cv2.imwrite(output_path, frame)
            if time.time() - last_log_write_time > LOG_WRITE_INTERVAL_SECONDS:
                write_logs_from_buffer()
                last_log_write_time = time.time()
            elapsed_time = time.time() - start_time
            sleep_time = max(0, PROCESSING_INTERVAL_SECONDS - elapsed_time)
            time.sleep(sleep_time)
    finally:
        print("\nExiting... Writing any remaining logs.")
        write_logs_from_buffer()
        for cap in caps.values():
            cap.release()
        print("System shut down gracefully.")

if __name__ == "__main__":
    main()