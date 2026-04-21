import cv2
import numpy as np
from ultralytics import YOLO

def create_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    # x, y, vx, vy
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    
    # --- KRİTİK AYARLAR ---
    # Q (Process Noise): Ne kadar düşükse, hayalet o kadar "çizgisel" gider.
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01 
    
    # R (Measurement Noise): Ne kadar yüksekse, kameradaki titremeleri o kadar umursamaz.
    # Titremeyi kesmek için bunu 0.1'den 0.5'e çektik.
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    
    # Hata kovaryansı başlangıcı
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
kalman = create_kalman()

last_valid_pos = None
ghost_counter = 0

print("Aslı, Stabilizasyon ayarları yapıldı. Şimdi daha 'ağırbaşlı' bir hayaletin var.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, stream=True, conf=0.6, verbose=False)
    detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # --- AYKIRI DEĞER KONTROLÜ ---
            # Eğer yeni nokta bir önceki noktadan 200 pikselden fazlaysa "glitch" kabul et
            if last_valid_pos is not None:
                dist = np.sqrt((cx - last_valid_pos[0])**2 + (cy - last_valid_pos[1])**2)
                if dist > 200: 
                    continue # Bu kareyi atla, çok saçma bir zıplama
            
            measured = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measured)
            
            last_valid_pos = (cx, cy)
            detected = True
            ghost_counter = 0
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            break

    # --- TAHMİN ---
    prediction = kalman.predict()
    px, py = int(prediction[0, 0]), int(prediction[1, 0])

    if not detected:
        ghost_counter += 1
        # Sadece 20 kare (yaklaşık 1 saniye) hayaleti izle, sonra vazgeç
        if ghost_counter < 20:
            cv2.circle(frame, (px, py), 15, (0, 255, 255), 2)
            cv2.putText(frame, "GHOSTING", (px + 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Hayalet çizgisi (Gittiği yönü gösterir)
            cv2.line(frame, (px, py), (px + int(prediction[2, 0]*10), py + int(prediction[3, 0]*10)), (0, 255, 255), 2)
        else:
            last_valid_pos = None # Tamamen kaybolduysa sıfırla
    else:
        # Kilitliyken tahmin noktasını (yeşil) kontrol et, kutunun tam ortasında olmalı
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    cv2.imshow("Ghost Follower v5 - Ultra Smooth", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()