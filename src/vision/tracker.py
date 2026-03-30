import numpy as np
from collections import OrderedDict

class KalmanFilter:
    """
    Basit 2D Kalman Filtresi: [x, y, vx, vy] durum modeli.
    Nesnenin merkezini ve anlık hızını takip eder.
    """
    def __init__(self, dt=0.133, process_variance=1e-4, measurement_variance=5.0):
        # dt ~= 1/7.5 fps (şartnameye göre 7.5 FPS)
        self.dt = dt
        self.state = np.zeros(4, dtype=np.float64)  # [x, y, vx, vy]
        self.P = np.eye(4) * 100.0

        # Geçiş matrisi
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)

        # Gözlem matrisi (sadece x, y ölçülür)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

        self.Q = np.eye(4) * process_variance
        self.R = np.eye(2) * measurement_variance

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def velocity(self) -> float:
        """Piksel/kare cinsinden hız büyüklüğü."""
        vx, vy = self.state[2], self.state[3]
        return float(np.sqrt(vx**2 + vy**2))


class CentroidTracker:
    """
    Kalman Filtreli Centroid Takipçisi.

    TEKNOFEST Görev 1 Şartnamesi:
    - Taşıtların hareketli/hareketsiz durumu belirlenmelidir.
    - Kamera hareketi sebebi ile sabit taşıtlar da hareket ediyormuş gibi görünebilir;
      bu durum ayırt edilebilmelidir.

    Strateji:
    - Optik akış tabanlı bir baseline (kamera hareketi kestirimi) olmadan
      saf piksel hız eşiği kullanıyoruz. Eşik değeri aşılırsa "Hareketli (1)" döndürür.
    - Kamera hareketi düzeltmesi istenirse optical flow eklenmeli (ileride geliştirilebilir).
    """

    MOTION_VELOCITY_THRESHOLD = 3.0   # piksel/kare, altında -> hareketsiz
    MAX_DISAPPEARED = 30              # kare sayısı
    MAX_DISTANCE = 120                # piksel, eşleme için

    def __init__(self):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.filters: OrderedDict[int, KalmanFilter] = OrderedDict()

    # ------------------------------------------------------------------
    # İç Yardımcılar
    # ------------------------------------------------------------------
    def _register(self, centroid: np.ndarray):
        kf = KalmanFilter()
        kf.state[:2] = centroid
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.filters[self.next_id] = kf
        self.next_id += 1

    def _deregister(self, oid: int):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.filters[oid]

    # ------------------------------------------------------------------
    # Ana Güncelleme
    # ------------------------------------------------------------------
    def update(self, rects: list) -> dict:
        """
        Karelerdeki bounding box listesi ile takipçiyi günceller.

        Args:
            rects: [(x1, y1, x2, y2), ...] formatında bounding box listesi.

        Returns:
            {object_id: centroid_np_array} dict.
        """
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                self.objects[oid] = self.filters[oid].predict().astype(int)
                if self.disappeared[oid] > self.MAX_DISAPPEARED:
                    self._deregister(oid)
            return self.objects

        # Yeni centroid'leri hesapla
        input_centroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in rects],
            dtype=int
        )

        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return self.objects

        # Mesafe matrisi
        object_ids = list(self.objects.keys())
        predictions = np.array([self.filters[oid].predict() for oid in object_ids])

        D = np.linalg.norm(
            predictions[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2
        )  # shape: (n_tracked, n_detected)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.MAX_DISTANCE:
                continue

            oid = object_ids[row]
            self.filters[oid].update(input_centroids[col].astype(float))
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Kaybolan nesneler
        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            self.objects[oid] = self.filters[oid].predict().astype(int)
            if self.disappeared[oid] > self.MAX_DISAPPEARED:
                self._deregister(oid)

        # Yeni nesneler
        for col in set(range(D.shape[1])) - used_cols:
            self._register(input_centroids[col])

        return self.objects

    # ------------------------------------------------------------------
    # Yardımcı Sorgular
    # ------------------------------------------------------------------
    def get_motion_status(self, object_id: int) -> str:
        """
        Şartnameye göre hareket durumu:
        "0" = Hareketsiz, "1" = Hareketli
        Takipçide yoksa "0" döner.
        """
        if object_id not in self.filters:
            return "0"
        velocity = self.filters[object_id].velocity
        return "1" if velocity > self.MOTION_VELOCITY_THRESHOLD else "0"

    def get_id_for_centroid(self, cx: int, cy: int) -> int:
        """
        En yakın takip ID'sini döndürür (tracker ile tespit eşleştirme için).
        """
        if not self.objects:
            return -1
        ids = list(self.objects.keys())
        centroids = np.array(list(self.objects.values()))
        dists = np.linalg.norm(centroids - np.array([cx, cy]), axis=1)
        return ids[int(np.argmin(dists))]
