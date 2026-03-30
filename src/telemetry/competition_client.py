import requests
import json
import time
import io
import numpy as np
import cv2

class CompetitionClient:
    """
    TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması - Sunucu İletişim İstemcisi.

    Şartname Bölüm 8 - Sunucu İle Bağlantı:
    - Tüm haberleşme REST API + JSON formatında.
    - Sunucu adresi: örn. http://127.0.0.25:5000
    - Kare başına TAM OLARAK 1 sonuç paket gönderilmelidir.
    - Sonuç gönderilmeden sonraki kare isteğinde bulunulamaz.

    JSON Formatı (Şekil 17):
    {
        "user":     "<kullanıcı url>",
        "frame":    "<kare url>",
        "detected_objects": [
            {
                "cls":             "0"|"1"|"2"|"3",
                "landing_status":  "-1"|"0"|"1",
                "motion_status":   "-1"|"0"|"1",
                "top_left_x":      int,
                "top_left_y":      int,
                "bottom_right_x":  int,
                "bottom_right_y":  int
            }, ...
        ],
        "detected_translations": [
            {
                "translation_x": float,
                "translation_y": float,
                "translation_z": float
            }
        ],
        "detected_undefined_objects": [
            {
                "object_id":      str,
                "top_left_x":     int,
                "top_left_y":     int,
                "bottom_right_x": int,
                "bottom_right_y": int
            }, ...
        ]
    }
    """

    def __init__(self, base_url: str = "http://127.0.0.1:5000", username: str = "team_skyguard"):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self._last_frame_data: dict = {}
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        print(f"[Client] Başlatıldı: {self.base_url} | Kullanıcı: {self.username}")

    # ------------------------------------------------------------------
    # Kare Al
    # ------------------------------------------------------------------
    def get_frame(self) -> dict:
        """
        Sunucudan sonraki video karesini ve telemetri verisini alır.
        Şartname Bölüm 8: url, image_url, video_name, session,
                          translation_x/y/z, gps_health_status döndürür.

        Returns:
            dict: Kare verisi. Hata durumunda boş dict.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/frame", timeout=10)
            if resp.status_code == 200:
                self._last_frame_data = resp.json()
                return self._last_frame_data
            else:
                print(f"[Client] Kare alınamadı: HTTP {resp.status_code}")
                return {}
        except requests.RequestException as e:
            print(f"[Client] Bağlantı hatası (get_frame): {e}")
            return {}

    # ------------------------------------------------------------------
    # Görüntü İndir
    # ------------------------------------------------------------------
    def download_image(self, image_url: str) -> np.ndarray:
        """
        Sunucudan görüntüyü indirir ve OpenCV formatına çevirir.

        Args:
            image_url (str): Kare görseli URL'i.

        Returns:
            np.ndarray: BGR formatında görüntü. Başarısız olursa None.
        """
        try:
            resp = self._session.get(image_url, timeout=15, stream=True)
            if resp.status_code == 200:
                img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
                frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                return frame
            else:
                print(f"[Client] Görüntü indirilemedi: HTTP {resp.status_code}")
                return None
        except requests.RequestException as e:
            print(f"[Client] Bağlantı hatası (download_image): {e}")
            return None

    # ------------------------------------------------------------------
    # Referans Nesneleri Al (Görev 3)
    # ------------------------------------------------------------------
    def get_reference_objects(self) -> list:
        """
        Oturum başında tanımsız nesne referanslarını alır (Görev 3).

        Returns:
            list: [{object_id, image_url}, ...] formatında liste.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/reference_objects", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[Client] {len(data)} referans nesne alındı.")
                return data
            else:
                print(f"[Client] Referans nesneler alınamadı: HTTP {resp.status_code}")
                return []
        except requests.RequestException as e:
            print(f"[Client] Bağlantı hatası (get_reference_objects): {e}")
            return []

    # ------------------------------------------------------------------
    # Sonuç Gönder
    # ------------------------------------------------------------------
    def send_results(
        self,
        detections: list,
        translation: dict,
        undefined_objects: list = None,
    ) -> bool:
        """
        Tespit ve pozisyon sonuçlarını sunucuya gönderir.

        Args:
            detections (list): detector.py çıktısı (cls, landing_status,
                               motion_status, bounding box).
            translation (dict): {translation_x, translation_y, translation_z} metre.
            undefined_objects (list): matcher.py çıktısı, Görev 3.

        Returns:
            bool: Başarı durumu.
        """
        if not self._last_frame_data:
            print("[Client] Hata: Gönderilecek ilişkili kare verisi yok.")
            return False

        # Tespit listesini sadece API alanlarıyla temizle
        clean_detections = []
        for d in detections:
            clean_detections.append({
                "cls":             str(d["cls"]),
                "landing_status":  str(d.get("landing_status", "-1")),
                "motion_status":   str(d.get("motion_status", "-1")),
                "top_left_x":      int(d["top_left_x"]),
                "top_left_y":      int(d["top_left_y"]),
                "bottom_right_x":  int(d["bottom_right_x"]),
                "bottom_right_y":  int(d["bottom_right_y"]),
            })

        payload = {
            "user":  f"{self.base_url}/api/users/{self.username}/",
            "frame": self._last_frame_data.get("url", ""),
            "detected_objects": clean_detections,
            "detected_translations": [
                {
                    "translation_x": float(translation.get("translation_x", 0.0)),
                    "translation_y": float(translation.get("translation_y", 0.0)),
                    "translation_z": float(translation.get("translation_z", 0.0)),
                }
            ],
            "detected_undefined_objects": undefined_objects or [],
        }

        try:
            resp = self._session.post(
                f"{self.base_url}/api/results",
                data=json.dumps(payload),
                timeout=10
            )
            if resp.status_code in [200, 201]:
                return True
            else:
                print(f"[Client] Sonuç gönderilemedi: HTTP {resp.status_code} | {resp.text[:200]}")
                return False
        except requests.RequestException as e:
            print(f"[Client] Bağlantı hatası (send_results): {e}")
            return False

    # ------------------------------------------------------------------
    # Test / Bağlantı Kontrolü
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Sunucuya basit bağlantı testi."""
        try:
            resp = self._session.get(f"{self.base_url}/api/", timeout=5)
            return resp.status_code < 500
        except requests.RequestException:
            return False


if __name__ == "__main__":
    # Bağlantı testi
    client = CompetitionClient()
    frame = client.get_frame()
    if frame:
        print(f"Kare alındı: {frame.get('video_name')} | {frame.get('image_url')}")
