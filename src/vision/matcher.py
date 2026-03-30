import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class ObjectMatcher:
    """
    Tanımsız Nesne Eşleştirme Modülü - TEKNOFEST 2026 Görev 3.

    Şartname Özeti (Bölüm 2.3):
    - Oturum başında referans nesneler (görseller) paylaşılır.
    - Uçuş sırasında bu referans nesneler karelerde aranır.
    - Zorluklar: farklı kamera (termal/RGB), farklı açı/irtifa, uydu görüntüsü,
                 yer seviyesi görseller, çeşitli görüntü işleme filtreleri.
    - Bazı referans nesneler oturumda hiç görünmeyebilir.

    Strateji:
    - SIFT öznitelikleri: ölçek + rotasyon değişmezliği sağlar.
    - Lowe's Ratio Test: sahte eşleşmeleri eler.
    - RANSAC Homografi: perspektif değişikliklerine dayanıklı konum hesabı.
    - Minimum eşleşme sayısı kontrolü ile yanlış pozitif oranı düşürülür.
    """

    MIN_MATCH_COUNT  = 10    # Homografi için minimum iyi eşleşme sayısı
    LOWE_RATIO       = 0.72  # Lowe's ratio test eşiği
    RANSAC_THRESHOLD = 5.0   # Homografi RANSAC piksel eşiği

    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04)
        self.bf   = cv2.BFMatcher(cv2.NORM_L2)
        # {object_id: (kps, des, (h, w))}
        self.reference_objects: Dict[str, Tuple] = {}
        print("[ObjectMatcher] SIFT tabanlı eşleştirici hazır.")

    # ------------------------------------------------------------------
    # Referans Nesne Kayıt
    # ------------------------------------------------------------------
    def add_reference_object(self, object_id: str, image: np.ndarray):
        """
        Oturum başında sunucudan alınan referans nesne görüntüsünü kaydeder.

        Args:
            object_id (str): Sunucudan gelen benzersiz nesne ID'si.
            image (np.ndarray): BGR veya gri tonlamalı referans görseli.
        """
        if image is None or image.size == 0:
            print(f"[ObjectMatcher] Uyarı: '{object_id}' için geçersiz görüntü.")
            return

        # BGR → Gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Kontrast iyileştirme (termal/karanlık görseller için CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        kps, des = self.sift.detectAndCompute(gray, None)

        if des is None or len(kps) < 5:
            print(f"[ObjectMatcher] Uyarı: '{object_id}' için yeterli öznitelik bulunamadı.")
            return

        self.reference_objects[object_id] = (kps, des, gray.shape[:2])
        print(f"[ObjectMatcher] Referans eklendi: '{object_id}' | {len(kps)} öznitelik.")

    def clear_references(self):
        """Oturum sona erince referansları temizler."""
        self.reference_objects.clear()

    # ------------------------------------------------------------------
    # Eşleştirme
    # ------------------------------------------------------------------
    def match(self, frame: np.ndarray) -> List[Dict]:
        """
        Kaydedilmiş tüm referans nesneleri verilen karede arar.

        Args:
            frame (np.ndarray): BGR formatında anlık kare.

        Returns:
            list: Eşleşen nesneler için dicts:
                  {object_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y}
                  Şartneme Şekil 17: detected_undefined_objects formatı.
        """
        if not self.reference_objects:
            return []

        # Kare ön işleme
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(gray_frame)

        kps_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)

        results = []

        if des_frame is None or len(kps_frame) < 5:
            return results

        for obj_id, (kps_ref, des_ref, shape_ref) in self.reference_objects.items():
            box = self._find_object(kps_ref, des_ref, shape_ref,
                                    kps_frame, des_frame, gray_frame.shape)
            if box is not None:
                x1, y1, x2, y2 = box
                results.append({
                    "object_id": obj_id,
                    "top_left_x":     int(x1),
                    "top_left_y":     int(y1),
                    "bottom_right_x": int(x2),
                    "bottom_right_y": int(y2),
                })
                print(f"[ObjectMatcher] Eşleşme: '{obj_id}' @ ({x1},{y1})-({x2},{y2})")

        return results

    # ------------------------------------------------------------------
    # İç Yardımcı
    # ------------------------------------------------------------------
    def _find_object(
        self,
        kps_ref, des_ref, shape_ref,
        kps_frame, des_frame, frame_shape
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Referans özniteliklerini kare öznitelikleriyle eşleştirip
        bounding box döndürür. Bulunamazsa None döner.
        """
        # kNN eşleştirme (k=2 → Lowe's ratio)
        try:
            matches = self.bf.knnMatch(des_ref, des_frame, k=2)
        except cv2.error:
            return None

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.LOWE_RATIO * n.distance:
                good.append(m)

        if len(good) < self.MIN_MATCH_COUNT:
            return None

        src_pts = np.float32([kps_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_THRESHOLD)
        if M is None:
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.MIN_MATCH_COUNT:
            return None

        # Referans nesnenin köşelerini kareye dönüştür
        h_ref, w_ref = shape_ref
        corners = np.float32([[0, 0], [0, h_ref - 1],
                               [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(corners, M)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(dst_corners)

        # Kare sınırları içinde olduğunu doğrula
        fh, fw = frame_shape[:2]
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(fw - 1, x + bw)
        y2 = min(fh - 1, y + bh)

        if (x2 - x) < 5 or (y2 - y) < 5:  # Çok küçük kutu → anlamsız
            return None

        return (x, y, x2, y2)
