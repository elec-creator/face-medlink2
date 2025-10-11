from deepface import DeepFace
import cv2
import time
import serial
import numpy as np

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
COOLDOWN = 2.0
RESET_FRAMES = 10

# ==== serial ====
try:
    esp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] Serial terbuka di {SERIAL_PORT}")
except Exception as e:
    print(f"[WARNING] Tidak bisa membuka serial: {e}")
    esp = None

# ==== haarcascade (cepat banget) ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_state = "stop"
last_name = ""
last_detect_time = 0
absence_counter = 0

# ==== open camera ====
cap = cv2.VideoCapture(0)

print("[INFO] Mulai deteksi real-time...")
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

        # identifikasi hanya kalau cukup waktu berlalu
        if time.time() - last_detect_time > COOLDOWN:
            try:
                result = DeepFace.find(img_path=roi, db_path="Data/",
                                       model_name="Facenet512",
                                       distance_metric="euclidean_l2",
                                       enforce_detection=False)
                if len(result) > 0 and not result[0].empty:
                    name = result[0]['identity'][0].split('/')[1]
                    last_name = name
                    if last_state != "start":
                        if esp:
                            esp.write(f"start:{name}\n".encode())
                        print(f"[{time.strftime('%H:%M:%S')}] ✅ Kirim start:{name}")
                        last_state = "start"
                last_detect_time = time.time()
            except Exception as e:
                print("[WARN] Gagal identifikasi:", e)

        cv2.putText(frame, last_name if last_name else "Scanning...", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # kalau tidak ada wajah
    if not detected:
        absence_counter += 1
        if absence_counter > RESET_FRAMES and last_state != "stop":
            if esp:
                esp.write(b"stop\n")
            print(f"[{time.strftime('%H:%M:%S')}] ⛔ Kirim stop (wajah hilang)")
            last_state = "stop"
            last_name = ""
            absence_counter = 0
    else:
        absence_counter = 0

    cv2.putText(frame, f"State: {last_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if esp:
    esp.close()
    print("[INFO] Serial ditutup.")
