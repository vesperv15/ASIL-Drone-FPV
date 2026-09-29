# The Ghost Follower

**The Ghost Follower**, YOLOv8 nesne tespiti modelini matematiksel filtreleme ve konum tahmin algoritmalarıyla (EMA ve Kalman Filtresi) birleştiren bir bilgisayarla görme (Computer Vision) ve nesne takip projesidir.

Proje; kameradan alınan canlı görüntü üzerinde nesne tespitinin ötesine geçerek, nesnenin hareket yönünü, hızını ve gelecekte bulunacağı konumları "Hayalet (Ghost)" noktaları ve çizgileriyle tahmin eder.

---

## Özellikler

- **Gerçek Zamanlı Nesne Tespiti:** Ultralytics YOLOv8 Nano (`yolov8n.pt`) modeli ile yüksek FPS değerlerinde tespit.
- **Hız Tahmini ve Gelecek Konum Projeksiyonu:** Hareket yönüne bağlı olarak nesnenin 15 kare sonrasını tahmin etme.
- **Kalman Filtresi İle Yumuşatma (Ultra Smooth Tracking):** Kamera ve tespit titremelerini (jitter) engelleyen 4-durumlu (x, y, vx, vy) Kalman Filtresi uygulaması.
- **Aykırı Değer (Glitch) Engelleyici:** Aniden sıçrayan hatalı tespitleri filtreleyen piksel bazlı uzaklık kontrolü.
- **Geçici Kayıp (Occlusion) Toleransı:** Nesne kadrajdan kısa süreli çıktığında dahi konumunu 20 kare boyunca tahmin etmeye devam etme.

---
## Algoritma detayları
 1. EMA (Exponential Moving Average) — ghost_tracker.pyNesnenin merkez koordinatlarındaki ($c_x, c_y$) anlık değişim $v_x$ ve $v_y$ hız vektörlerine dönüştürülür:
$$\mathbf{v}_{yeni} = \alpha \cdot \mathbf{v}_{anlik} + (1 - \alpha) \cdot \mathbf{v}_{eski}$$
Buradaki $\alpha = 0.4$ katsayısı titremeyi azaltırken gecikmeyi minimumda tutar.
 2. Kalman Filtresi — kalman_follower.py
OpenCV KalmanFilter(4, 2) yapısı kullanılır:
 - Durum Vektörü: $[x, y, v_x, v_y]^T$
 - Ölçüm Vektörü: $[x, y]^T$
 - Gürültü Ayarları: processNoiseCov $0.01$ seviyesinde tutularak kararlı yön çizgisi elde edilir, measurementNoiseCov $0.5$ değerine çekilerek kamera titremeleri sönümlenir.


## Proje Yapısı

```text
.
├── vision_test.py      # 1. Gün: Temel YOLOv8 testi ve canlı FPS ölçümü
├── ghost_tracker.py    # 2. Gün: Üstel Hareketli Ortalama (EMA) ile hayalet takibi
├── kalman_follower.py  # v5: Kalman Filtresi ve stabilizasyon destekli gelişmiş takip
├── yolov8n.pt          # YOLOv8 Nano ağırlık dosyası
└── .gitignore          # Sanal ortam (.venv) yapılandırması
