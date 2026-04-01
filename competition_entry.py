"""
competition_entry.py — TEKNOFEST 2026 Havacılıkta Yapay Zeka
Ana yarışma döngüsü.

Şartneme (sartname_teknik.txt) Özeti:
- Görev 1: Nesne tespiti (Taşıt=0, İnsan=1, UAP=2, UAİ=3) + hareket + iniş durumu
- Görev 2: GPS sağlıklı değilse visual odometry ile pozisyon kestir
- Görev 3: Oturum başında verilen referans nesneleri anlık karelerde bul
- Her kare için TAM OLARAK 1 sonuç paketi gönder, sonuç göndermeden sonraki kareyi isteyemezsin.
- cls, landing_status, motion_status: string tipinde ("-1", "0", "1")
- IoU eşiği: 0.5 (Şartname Bölüm 9.1)

Puanlama Ağırlıkları (Tablo 7):
- Görev 1: %25, Görev 2: %40, Görev 3: %25, Final Raporu: %5, Sunum: %5
"""

import cv2
import numpy as np
import time
import requests

from src.vision.detector import ObjectDetector
from src.vision.tracker import CentroidTracker
from src.vision.odometry import VisualOdometry
from src.vision.matcher import ObjectMatcher
from src.telemetry.competition_client import CompetitionClient


# ======================================================================
# İniş Uygunluk Kontrolü - Şartname Bölüm 2.1.2
# ======================================================================

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    İki bounding box arasındaki IoU değerini hesaplar.
    box: (x1, y1, x2, y2)
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def boxes_overlap(box_a: tuple, box_b: tuple) -> bool:
    """
    İki bounding box'ın örtüşüp örtüşmediğini kontrol eder.
    (IoU > 0 veya biri diğerinin içindeyse True)
    """
    return compute_iou(box_a, box_b) > 0.0


def check_landing_status(uap_uai_det: dict, all_detections: list, frame_w: int, frame_h: int) -> str:
    """
    UAP/UAİ alanının iniş uygunluğunu belirler.

    Şartname Kuralları:
    1. Alanın tamamı görüntü içinde olmalı → sınır piksellerine kesmemeli.
    2. Alan üzerinde herhangi bir nesne (tespit edilmiş veya değil) olmamalı.
       Pratik: Tespit edilen insan/taşıt/bilinmeyen ile çakışma → Uygun Değil.

    Returns:
        "0": Uygun Değil
        "1": Uygun
    """
    x1 = uap_uai_det['_raw_x1']
    y1 = uap_uai_det['_raw_y1']
    x2 = uap_uai_det['_raw_x2']
    y2 = uap_uai_det['_raw_y2']

    # Kural 1: Alanın tamamı kare içinde mi?
    if x1 < 0 or y1 < 0 or x2 >= frame_w or y2 >= frame_h:
        return "0"  # Bir kenar kısmen dışarıda → Uygun Değil

    # Kural 2: Çakışan nesne var mı?
    uap_box = (x1, y1, x2, y2)
    for other in all_detections:
        # Kendisiyle ve diğer UAP/UAİ alanlarıyla karşılaştırma
        if other is uap_uai_det:
            continue
        # Şartnameye göre üstünde herhangi bir cisim → Uygun Değil
        other_box = (
            other['top_left_x'], other['top_left_y'],
            other['bottom_right_x'], other['bottom_right_y']
        )
        if boxes_overlap(uap_box, other_box):
            return "0"

    return "1"


# ======================================================================
# Hareket Durumu - Şartname Bölüm 2.1.1
# ======================================================================

def enrich_with_motion_and_landing(
    detections: list,
    tracker: CentroidTracker,
    frame_w: int,
    frame_h: int,
    camera_shift: tuple = (0, 0)
) -> list:
    """
    Tespitleri tracker sonuçlarıyla zenginleştirir:
    - Taşıt (cls=0): motion_status tracker'dan alınır.
    - UAP/UAİ (cls=2/3): landing_status hesaplanır.
    - İnsan (cls=1): motion_status = "-1", landing_status = "-1"

    Args:
        detections: detector.py çıktısı.
        tracker: Güncellenmiş CentroidTracker.
        frame_w, frame_h: Kare boyutları.
        camera_shift: (dx, dy) kamera hareketi.

    Returns:
        Zenginleştirilmiş tespit listesi.
    """
    # Tüm tespitlerin kutu listesi (tracker için)
    rects = []
    for d in detections:
        rects.append((d['top_left_x'], d['top_left_y'],
                      d['bottom_right_x'], d['bottom_right_y']))

    tracker.update(rects, camera_shift=camera_shift)

    for d in detections:
        cls = d['cls']

        if cls == '0':  # Taşıt → hareket durumu
            cx = (d['top_left_x'] + d['bottom_right_x']) // 2
            cy = (d['top_left_y'] + d['bottom_right_y']) // 2
            obj_id = tracker.get_id_for_centroid(cx, cy)
            d['motion_status'] = tracker.get_motion_status(obj_id)
            d['landing_status'] = "-1"

        elif cls == '1':  # İnsan
            d['motion_status'] = "-1"
            d['landing_status'] = "-1"

        elif cls in ['2', '3']:  # UAP veya UAİ
            d['motion_status'] = "-1"
            d['landing_status'] = check_landing_status(d, detections, frame_w, frame_h)

    return detections


# ======================================================================
# Ana Döngü
# ======================================================================

def competition_loop(server_url: str = "http://127.0.0.1:5000", username: str = "team_skyguard"):
    """
    TEKNOFEST 2026 yarışma ana döngüsü.
    Şartname Bölüm 8'e uygun sıralı istek-cevap mimarisi.
    """
    print("=" * 60)
    print("  TEKNOFEST 2026 - Havacılıkta Yapay Zeka")
    print("  SkyGuard AI - Yarışma İstemcisi Başlatılıyor...")
    print("=" * 60)

    # Bileşenleri başlat
    client   = CompetitionClient(base_url=server_url, username=username)
    detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.35)
    tracker  = CentroidTracker()
    vo       = VisualOdometry()
    matcher  = ObjectMatcher()

    # ------------------------------------------------------------------
    # Görev 3: Oturum başında referans nesneleri al
    # ------------------------------------------------------------------
    print("\n[INIT] Görev 3 referans nesneleri alınıyor...")
    ref_objects = client.get_reference_objects()
    for ref in ref_objects:
        obj_id   = ref.get("object_id") or ref.get("id", "unknown")
        img_url  = ref.get("image_url")
        if img_url:
            ref_img = client.download_image(img_url)
            if ref_img is not None:
                matcher.add_reference_object(str(obj_id), ref_img)

    print(f"[INIT] {len(matcher.reference_objects)} referans nesne yüklendi.")
    print("[INIT] Kare döngüsü başlıyor...\n")

    frame_count    = 0
    success_count  = 0
    fail_count     = 0
    gps_was_healthy = True  # GPS geçiş takibi

    try:
        while True:
            t_start = time.time()

            # ----------------------------------------------------------
            # 1. Kare ve telemetri al
            # ----------------------------------------------------------
            frame_data = client.get_frame()
            if not frame_data:
                print("[LOOP] Kare alınamadı, 1 saniye bekleniyor...")
                time.sleep(1)
                continue

            image_url   = frame_data.get('image_url', '')
            video_name  = frame_data.get('video_name', '?')
            gps_health  = int(frame_data.get('gps_health_status', 1))
            gps_x       = float(frame_data.get('translation_x', 0.0))
            gps_y       = float(frame_data.get('translation_y', 0.0))
            gps_z       = float(frame_data.get('translation_z', 0.0))

            # ----------------------------------------------------------
            # 2. Görüntüyü indir
            # ----------------------------------------------------------
            frame = client.download_image(image_url)
            if frame is None:
                # Görüntü alınamadı → boş sonuç yolla (kare atlamama kuralı)
                client.send_results([], {"translation_x": gps_x,
                                         "translation_y": gps_y,
                                         "translation_z": gps_z})
                frame_count += 1
                fail_count  += 1
                continue

            frame_h, frame_w = frame.shape[:2]
            frame_count += 1

            # ----------------------------------------------------------
            # 3. Görev 2: Pozisyon Kestirimi (Önce yapıyoruz ki kamera kaymasını alalım)
            # ----------------------------------------------------------
            if gps_health == 1:
                # GPS sağlıklı: sunucu değerini kullan VE visual odometry'yi hizala
                translation = {"translation_x": gps_x,
                               "translation_y": gps_y,
                               "translation_z": gps_z}
                vo.align_with_gps(gps_x, gps_y, gps_z)
                # VO'yu güncelle (GPS açık olsa da devam et; geçiş anında kaymasız)
                vo.update(frame, altitude_m=abs(gps_z))
                gps_was_healthy = True

            else:
                # GPS sağlıksız: visual odometry tahmini
                if gps_was_healthy:
                    # GPS yeni kesildi → sıfırla ve hizala
                    vo.reset()
                    vo.align_with_gps(gps_x, gps_y, gps_z)
                    gps_was_healthy = False

                pos = vo.update(frame)
                corrected = vo.get_corrected_position()
                translation = {
                    "translation_x": float(corrected[0]),
                    "translation_y": float(corrected[1]),
                    "translation_z": float(corrected[2]),
                }

            # Kamera kaymasını al (dx, dy)
            camera_shift = (float(vo.last_shift[0]), float(vo.last_shift[1]))

            # ----------------------------------------------------------
            # 4. Görev 1: Nesne Tespiti
            # ----------------------------------------------------------
            detections, _ = detector.detect(frame)

            # Hareket ve iniş durumlarını zenginleştir (kamera kayması dahil)
            detections = enrich_with_motion_and_landing(
                detections, tracker, frame_w, frame_h, camera_shift=camera_shift
            )

            # ----------------------------------------------------------
            # 5. Görev 3: Tanımsız Nesne Eşleme
            # ----------------------------------------------------------
            undefined_objects = matcher.match(frame)

            # ----------------------------------------------------------
            # 6. Sonuçları Gönder
            # ----------------------------------------------------------
            ok = client.send_results(detections, translation, undefined_objects)

            if ok:
                success_count += 1
            else:
                fail_count += 1

            t_elapsed = time.time() - t_start
            status_str = "OK" if ok else "ERR"
            gps_str    = "GPS:OK" if gps_health == 1 else "GPS:XX"
            n_obj      = len(detections)
            n_undef    = len(undefined_objects)

            print(
                f"[{status_str}] Kare {frame_count:4d} | {video_name} | "
                f"{gps_str} | Nesne:{n_obj:2d} | Ref:{n_undef} | "
                f"t={t_elapsed*1000:.0f}ms | "
                f"OK:{success_count} ERR:{fail_count}",
                end='\r'
            )

    except KeyboardInterrupt:
        print("\n\n[STOP] Yarışma döngüsü durduruldu.")
        print(f"  Toplam kare: {frame_count}")
        print(f"  Başarılı  : {success_count}")
        print(f"  Başarısız : {fail_count}")
    finally:
        print("Temizleniyor...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 Yarışma İstemcisi")
    parser.add_argument(
        "--server",  type=str, default="http://127.0.0.1:5000",
        help="Yarışma sunucusu URL'i (örn: http://192.168.1.100:5000)"
    )
    parser.add_argument(
        "--user",    type=str, default="team_skyguard",
        help="Takım kullanıcı adı"
    )
    args = parser.parse_args()

    competition_loop(server_url=args.server, username=args.user)
