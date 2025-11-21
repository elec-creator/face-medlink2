from deepface import DeepFace
import cv2
import time
import requests
import numpy as np

COOLDOWN = 2.0
RESET_FRAMES = 10

API_URL = "http://smartmush.my.id/smartmush/api_medlink/set_state.php"

# ==== haarcascade ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_state = "stop"
last_name = ""
last_detect_time = 0
absence_counter = 0

# ==== open camera ====
cap = cv2.VideoCapture(0)

print("[INFO] Mulai deteksi real-time...")

def kirim_api(state, name=None):
    """Kirim state ke API (start/stop)."""
    try:
        params = {"state": state}
        if name:
            params["name"] = name

        r = requests.get(API_URL, params=params, timeout=2)
        print("[API]", r.text)
    except Exception as e:
        print("[API ERROR]", e)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    detected = False

    for (x, y, w, h) in faces:
        detected = True
        roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # identifikasi ketika cooldown selesai
        if time.time() - last_detect_time > COOLDOWN:
            try:
                result = DeepFace.find(img_path=roi, db_path="Data/",
                                       model_name="Facenet512",
                                       distance_metric="euclidean_l2",
                                       enforce_detection=False)

                if len(result) > 0 and not result[0].empty:
                    name = result[0]['identity'][0].split('/')[1]
                    last_name = name

                    # === Kirim START ke API ===
                    if last_state != "start":
                        kirim_api("start", name)
                        print(f"[{time.strftime('%H:%M:%S')}] ✅ START: {name}")
                        last_state = "start"

                last_detect_time = time.time()
            except Exception as e:
                print("[WARN] Gagal identifikasi:", e)

        cv2.putText(frame, last_name if last_name else "Scanning...", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # === Jika wajah tidak terdeteksi ===
    # Python tidak kirim STOP
    if not detected:
        absence_counter += 1
        if absence_counter > RESET_FRAMES:
            last_state = "stop"
            last_name = ""

    else:
        absence_counter = 0

    cv2.putText(frame, f"State: {last_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Sistem ditutup.")
