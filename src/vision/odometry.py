import cv2
import numpy as np

class VisualOdometry:
    """
    Görsel Odometri Modülü - TEKNOFEST 2026 Görev 2.

    Şartname Özeti (Bölüm 2.2):
    - GPS sağlık değeri (gps_health_status) 1 ise: sunucu verisini kullanabilirsin.
    - GPS sağlık değeri 0 ise: kendi kestirdiğin pozisyonu göndermelisin.
    - İlk pozisyon x0=0.00, y0=0.00, z0=0.00.
    - Kestirilen pozisyon metre cinsinden, referans koordinat sistemine göre.
    - FPS: 7.5 (dt ≈ 0.133 saniye).

    Algoritma:
    1. ORB öznitelik tespiti + BFMatcher ile ardışık kareler arasında eşleme.
    2. RANSAC ile Essential Matrix hesabı.
    3. recoverPose ile göreli R, t hesabı.
    4. GPS sağlığını kamera parametreleriyle ölçeklendirme (z = irtifa).
    5. Kümülatif konum güncelleme.
    """

    def __init__(self, camera_matrix=None, baseline_scale: float = 1.0):
        """
        Args:
            camera_matrix (np.ndarray | None): 3x3 kamera iç parametresi.
                Yoksa Full-HD için makul bir varsayılan kullanılır.
            baseline_scale (float): t vektörünü metreye ölçeklemek için.
                İrtifa verileri geldiğinde dinamik olarak güncellenebilir.
        """
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Kamera matrisi (Full HD 1920x1080 için yaklaşık değer)
        if camera_matrix is None:
            self.K = np.array([
                [1400.0,    0.0, 960.0],
                [   0.0, 1400.0, 540.0],
                [   0.0,    0.0,   1.0]
            ], dtype=np.float64)
        else:
            self.K = np.array(camera_matrix, dtype=np.float64)

        self.baseline_scale = baseline_scale

        # Durum değişkenleri
        self.current_pos = np.zeros(3, dtype=np.float64)  # [x, y, z] metre
        self.rotation    = np.eye(3, dtype=np.float64)    # Kümülatif dönme

        # Bir önceki kareye ait bilgiler
        self.prev_gray = None
        self.prev_kps  = None
        self.prev_des  = None

        # GPS sağlığından hesaplanan referans pozisyon
        self._gps_reference_pos = np.zeros(3, dtype=np.float64)
        self._last_gps_pos      = np.zeros(3, dtype=np.float64)
        self._gps_was_healthy   = True
        self.last_shift         = np.zeros(2, dtype=np.float64)  # [dx, dy] pixels

        print("[VisualOdometry] Başlatıldı. K matrisi:")
        print(self.K)

    # ------------------------------------------------------------------
    # Ana Güncelleme
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, altitude_m: float = None) -> np.ndarray:
        """
        Bir sonraki kar ile pozisyon tahminini günceller.

        Args:
            frame (np.ndarray): BGR formatında kare.
            altitude_m (float | None): GPS'ten irtifa (metre). Ölçek için.

        Returns:
            np.ndarray: [x, y, z] metre cinsinden kümülatif pozisyon.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or des is None or self.prev_des is None:
            self._store_frame(gray, kps, des)
            return self.current_pos.copy()

        # Öznitelik eşleme
        matches = self.bf.match(self.prev_des, des)
        if len(matches) < 8:
            self._store_frame(gray, kps, des)
            return self.current_pos.copy()

        matches = sorted(matches, key=lambda m: m.distance)[:200]

        pts1 = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[m.trainIdx].pt          for m in matches])

        # Essential Matrix -> R, t
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            self._store_frame(gray, kps, des)
            return self.current_pos.copy()

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        # Piksel kayması hesapla (RANSAC inlier'ları kullanarak)
        mask_bool = mask.ravel() == 1
        if mask_bool.any():
            self.last_shift = np.mean(pts2[mask_bool] - pts1[mask_bool], axis=0)
        else:
            self.last_shift = np.zeros(2)

        # Ölçek: irtifa varsa kullan, yoksa sabit baseline_scale
        scale = self.baseline_scale
        if altitude_m is not None and altitude_m > 0.1:
            scale = altitude_m * 0.05  # Ampirik: irtifanın ~%5'i

        # Kümülatif pozisyon güncelle (dünya koordinat sistemi)
        self.current_pos += self.rotation @ (t.flatten() * scale)
        self.rotation = R @ self.rotation

        self._store_frame(gray, kps, des)
        return self.current_pos.copy()

    # ------------------------------------------------------------------
    # GPS Geçiş Yönetimi
    # ------------------------------------------------------------------
    def align_with_gps(self, gps_x: float, gps_y: float, gps_z: float):
        """
        GPS sağlıklıyken görsel odometreyi GPS referansıyla hizalar.
        Bu sayede GPS tekrar sağlıksız olduğunda referans kaymasız devam edilir.
        """
        self._last_gps_pos = np.array([gps_x, gps_y, gps_z], dtype=np.float64)
        # Mevcut visual pos ile GPS arasındaki kayma (drift)
        drift = self._last_gps_pos - self.current_pos
        self._gps_reference_pos = drift
        self._gps_was_healthy = True

    def get_corrected_position(self) -> np.ndarray:
        """
        GPS-visual kaymasını düzelterek tahmin edilen pozisyonu döndürür.
        """
        return self.current_pos + self._gps_reference_pos

    # ------------------------------------------------------------------
    # Sıfırlama
    # ------------------------------------------------------------------
    def reset(self):
        """GPS sağlığı geri geldi; sıfırlayıp hizala."""
        self.current_pos = np.zeros(3, dtype=np.float64)
        self.rotation    = np.eye(3, dtype=np.float64)
        self.prev_gray   = None
        self.prev_kps    = None
        self.prev_des    = None
        self._gps_reference_pos = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # İç Yardımcı
    # ------------------------------------------------------------------
    def _store_frame(self, gray, kps, des):
        self.prev_gray = gray
        self.prev_kps  = kps
        self.prev_des  = des
