"""
tests/test_competition_logic.py
Unit testler - TEKNOFEST 2026 şartnamesine uygunluk kontrolü.
Çalıştır: pytest tests/test_competition_logic.py -v

NOT: Bu testler ultralytics veya cv2 model gerektirmeden çalışır.
     Yalnızca saf Python mantık fonksiyonlarını test eder.
"""

import sys
import os
import types
import importlib

# Ultralytics yoksa mock'la (CI/test ortamında)
if "ultralytics" not in sys.modules:
    mock_ultra = types.ModuleType("ultralytics")
    class _FakeYOLO:
        def __init__(self, *a, **kw): self.names = {}
        def __call__(self, *a, **kw): return []
    mock_ultra.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = mock_ultra

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from competition_entry import compute_iou, boxes_overlap, check_landing_status


# ===========================================================================
# IoU Testleri
# ===========================================================================

class TestIoU:
    def test_identical_boxes(self):
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        # intersection = 50*50=2500, union = 10000+10000-2500=17500
        assert compute_iou(a, b) == pytest.approx(2500 / 17500, rel=1e-3)

    def test_iou_threshold_05(self):
        """Şartname: IoU eşiği 0.5 (Bölüm 9.1)"""
        a = (0, 0, 100, 100)
        b = (0, 0, 71, 100)  # % yaklaşık 71 x 100 → iou ~0.71
        assert compute_iou(a, b) > 0.5


# ===========================================================================
# Landing Status Testleri
# ===========================================================================

FRAME_W, FRAME_H = 1920, 1080

def make_det(x1, y1, x2, y2, cls="2"):
    """Test için örnek tespit dict üretir."""
    return {
        'cls': cls,
        'top_left_x':     x1, 'top_left_y':     y1,
        'bottom_right_x': x2, 'bottom_right_y': y2,
        '_raw_x1': x1, '_raw_y1': y1,
        '_raw_x2': x2, '_raw_y2': y2,
        'landing_status': '1', 'motion_status': '-1',
    }


class TestLandingStatus:
    def test_fully_inside_no_overlap_is_suitable(self):
        """Alanın tamamı kare içinde, üzerinde nesne yok → Uygun (1)"""
        uap = make_det(100, 100, 300, 300, cls="2")
        result = check_landing_status(uap, [uap], FRAME_W, FRAME_H)
        assert result == "1"

    def test_partial_outside_frame_is_unsuitable(self):
        """Alan sınırdan çıkıyor → Uygun Değil (0) — Şartname Bölüm 2.1.2"""
        uap = make_det(-10, 100, 300, 300, cls="2")  # x1 < 0
        result = check_landing_status(uap, [uap], FRAME_W, FRAME_H)
        assert result == "0"

    def test_person_on_landing_area_is_unsuitable(self):
        """Üzerinde insan var → Uygun Değil (0) — Şartname Bölüm 2.1.2"""
        uap    = make_det(100, 100, 400, 400, cls="2")
        person = make_det(150, 150, 250, 350, cls="1")  # Üstünde
        result = check_landing_status(uap, [uap, person], FRAME_W, FRAME_H)
        assert result == "0"

    def test_vehicle_next_to_area_no_overlap(self):
        """Yanındaki araç, alana çakışmıyor → Uygun (1)"""
        uap     = make_det(100, 100, 300, 300, cls="2")
        vehicle = make_det(400, 100, 600, 300, cls="0")  # Yanında, çakışmıyor
        result  = check_landing_status(uap, [uap, vehicle], FRAME_W, FRAME_H)
        assert result == "1"

    def test_top_right_corner_clipped(self):
        """Sağ kenardan kısmen dışarı çıkmış alan → Uygun Değil (0)"""
        uap = make_det(1800, 100, 1950, 400, cls="3")  # x2 > FRAME_W-1
        result = check_landing_status(uap, [uap], FRAME_W, FRAME_H)
        assert result == "0"


# ===========================================================================
# Tracker Testleri
# ===========================================================================

class TestCentroidTracker:
    def test_register_new_objects(self):
        from src.vision.tracker import CentroidTracker
        tracker = CentroidTracker()
        rects = [(0, 0, 100, 100), (200, 200, 400, 400)]
        objects = tracker.update(rects)
        assert len(objects) == 2

    def test_motion_status_stationary(self):
        from src.vision.tracker import CentroidTracker
        tracker = CentroidTracker()
        rect = [(100, 100, 200, 200)]
        for _ in range(10):
            tracker.update(rect)  # 10 kare boyunca sabit
        obj_id = list(tracker.objects.keys())[0]
        # Sabit nesne → hareketsiz
        assert tracker.get_motion_status(obj_id) == "0"

    def test_disappear_and_deregister(self):
        from src.vision.tracker import CentroidTracker
        tracker = CentroidTracker()
        tracker.MAX_DISAPPEARED = 5
        tracker.update([(100, 100, 200, 200)])
        assert len(tracker.objects) == 1
        for _ in range(10):
            tracker.update([])  # Boş kare
        assert len(tracker.objects) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
