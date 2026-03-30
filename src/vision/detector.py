import cv2
import numpy as np
from ultralytics import YOLO
import time

# Teknofest 2026 Şartnamesine göre Sınıf Tanımları (Tablo 2 & 4)
# Sınıf 0: Taşıt  -> Motorlu karayolu araçları, raylı araçlar, deniz araçları
# Sınıf 1: İnsan  -> Ayakta veya oturan kişiler
# Sınıf 2: UAP   -> Uçan Araba Park alanı (iniş durumu bildirmek zorunlu)
# Sınıf 3: UAİ   -> Uçan Ambulans İniş alanı (iniş durumu bildirmek zorunlu)

# YOLOv8 COCO sınıf adlarından Teknofest sınıf ID'lerine eşleme
COCO_TO_TEKNOFEST = {
    # Taşıtlar -> Sınıf 0
    'car': 0,
    'motorcycle': 0,
    'bus': 0,
    'truck': 0,
    'train': 0,
    'boat': 0,
    'bicycle': 0,       # scooter/bisiklet araç sayılır (sürücüsüz)
    # İnsan -> Sınıf 1
    'person': 1,
    # UAP ve UAİ alanları özel eğitimli model gerektirir; varsayılan COCO'da yok.
    # Bunlar özel model ile üretilir, class ismini model adından alırız:
    'uap': 2,
    'uai': 3,
    'parking': 2,
    'landing': 3,
}

class ObjectDetector:
    """
    TEKNOFEST 2026 Havacılıkta Yapay Zeka - Nesne Tespiti Modülü.
    Görev 1: Nesne tespiti (Taşıt, İnsan, UAP, UAİ) ve ilk hareket/landing status ataması.
    """

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.35, img_size=1280):
        """
        Args:
            model_path (str): YOLOv8 model dosyasının yolu.
                              Özel eğitimli model kullanılıyorsa onu belirtin.
            conf_threshold (float): Her tespit için minimum güven eşiği.
            img_size (int): Model girdi çözünürlüğü (yüksek çözünürlük = daha iyi tespit).
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        print(f"[Detector] Model '{model_path}' yüklendi. Eşik: {conf_threshold}")
        print(f"[Detector] Bilinen sınıflar: {list(self.model.names.values())}")

    def _map_class(self, class_name: str) -> int:
        """
        YOLO model sınıf adını Teknofest sınıf ID'sine çevirir.
        Bilinmeyen sınıf için -1 döndürür.
        """
        return COCO_TO_TEKNOFEST.get(class_name.lower(), -1)

    def detect(self, frame: np.ndarray):
        """
        Tek bir karedeki nesneleri tespit eder.

        Args:
            frame (np.ndarray): BGR formatında giriş görüntüsü.

        Returns:
            list: Tespit listesi. Her tespit şu anahtarları içerir:
                  'cls', 'confidence', 'top_left_x', 'top_left_y',
                  'bottom_right_x', 'bottom_right_y', 'landing_status', 'motion_status'
            np.ndarray: Görselleştirilmiş (annotated) kare.
        """
        results = self.model(frame, conf=self.conf_threshold,
                             imgsz=self.img_size, verbose=False)
        annotated_frame = frame.copy()
        detections = []

        frame_h, frame_w = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id_raw = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id_raw]

                tekno_cls = self._map_class(class_name)
                if tekno_cls == -1:
                    continue  # Teknofest dışı sınıf, atla

                # Şartnameye göre landing_status ve motion_status varsayılanları:
                # - Taşıt (0): iniş yok (-1), hareket tracker'dan gelecek
                # - İnsan (1): iniş yok (-1), hareket yok (-1)
                # - UAP (2) / UAİ (3): iniş durumu sonradan hesaplanacak, hareket yok (-1)
                landing_status = "-1"
                motion_status = "-1"

                if tekno_cls in [2, 3]:  # UAP veya UAİ
                    # Başlangıçta uygun varsay; landing check sonradan yapılacak
                    landing_status = "1"

                # Görüntü sınırlarını kontrol et (kısmi görünürlük için)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w - 1, x2)
                y2 = min(frame_h - 1, y2)

                detections.append({
                    'cls': str(tekno_cls),
                    'class_name': class_name,
                    'confidence': conf,
                    'top_left_x': x1,
                    'top_left_y': y1,
                    'bottom_right_x': x2,
                    'bottom_right_y': y2,
                    'landing_status': landing_status,
                    'motion_status': motion_status,
                    # Orijinal piksel sınırlarını sakla (landing check için)
                    '_raw_x1': int(box.xyxy[0][0].cpu().numpy()),
                    '_raw_y1': int(box.xyxy[0][1].cpu().numpy()),
                    '_raw_x2': int(box.xyxy[0][2].cpu().numpy()),
                    '_raw_y2': int(box.xyxy[0][3].cpu().numpy()),
                })

                # Görselleştirme
                color = (0, 255, 0) if tekno_cls == 0 else \
                        (255, 0, 0) if tekno_cls == 1 else \
                        (0, 165, 255) if tekno_cls == 2 else (0, 0, 255)
                label = f"T{tekno_cls}:{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        return detections, annotated_frame

    def run_on_video(self, source=0):
        """Webcam veya video dosyası üzerinde canlı test çalıştırır."""
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections, annotated_frame = self.detect(frame)
            cv2.imshow('SkyGuard - Nesne Tespiti', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run_on_video()
