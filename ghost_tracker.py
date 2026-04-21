import cv2
import numpy as np
from ultralytics import YOLO

model= YOLO('yolov8n.pt')
cap= cv2.VideoCapture(0)

prev_cx, prev_cy =0 , 0
v_x, v_y = 0, 0
alpha= 0.4 # 0.1 daha az titreme 0.4 geridn takip etmesin die
print("takip modu aktif")

while cap.isOpened():
    success, frame= cap.read()
    if not success: break
    
    results= model(frame, stream=True, conf= 0.5, verbose=False)
    target_detected=False
    
    for r in results:
        boxes= r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cx= (x1 + x2) // 2
            cy=(y1+ y2) // 2
            
            if prev_cx != 0:
                raw_dx = cx - prev_cx
                raw_dy= cy - prev_cy
                
                v_x =(alpha * raw_dx) + (1 - alpha) * v_x
                v_y =(alpha * raw_dy) +(1 - alpha) * v_y
                
                target_detected =True
            
            prev_cx, prev_cy = cx, cy
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            if target_detected:
                # 15 kare sonrasını tahmin et
                ghost_x = int(cx + v_x * 15)
                ghost_y = int(cy + v_y * 15)
                
                # Tahmin çizgisini çiz
                cv2.line(frame, (cx, cy), (ghost_x, ghost_y), (0, 255, 0), 2)
                cv2.circle(frame, (ghost_x, ghost_y), 8, (0, 255, 255), 2)
                cv2.putText(frame, "GHOST", (ghost_x + 10, ghost_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("The Ghost Follower - Day 2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()