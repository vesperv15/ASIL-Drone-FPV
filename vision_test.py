import cv2
import time
from ultralytics import YOLO

# 1. Modelimizi Yüklüyoruz (En hızlısı olan Nano modeli)
# İlk çalıştırdığında bu dosyayı otomatik internetten indirecek, panik yapma!
model = YOLO('yolov8n.pt')

# 2. Kamerayı Başlatıyoruz (Genelde 0 laptop kamerasıdır)
cap = cv2.VideoCapture(0)

prev_time = 0

print("Aslı, sistem başlıyor! Pencereyi kapatmak için 'q' tuşuna basabilirsin.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- YAPAY ZEKA ÇIKARIMI (Inference) ---
    # Stream=True yaparak hafızayı yormadan akış sağlıyoruz
    results = model(frame, stream=True, conf=0.5, verbase=False) # %50 güven skoru altındakileri görme

    # Sonuçları ekrana çizdiriyoruz
    for r in results:
        annotated_frame = r.plot()

    # --- MATEMATİKSEL FPS HESABI ---
    # FPS = 1 / (Şu anki zaman - Bir önceki kare zamanı)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # FPS bilgisini ekrana yazdıralım
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Ghost Follower - 1. Gun Testi", annotated_frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()