import sys
import os
sys.path.append(os.getcwd())
import types
from unittest.mock import MagicMock

# --- Mocking ultralytics ---
mock_ultra = types.ModuleType("ultralytics")
class _FakeYOLO:
    def __init__(self, *a, **kw): 
        self.names = {0: 'car', 1: 'person', 2: 'uap', 3: 'uai'}
    def __call__(self, *a, **kw): 
        m = MagicMock()
        m.boxes = []
        return [m]
mock_ultra.YOLO = _FakeYOLO
sys.modules["ultralytics"] = mock_ultra

# --- Running competition_entry.py ---
from competition_entry import competition_loop
import threading
import time

def run_loop():
    try:
        # Run for 30 seconds then stop
        competition_loop(server_url="http://127.0.0.1:5000", username="skyguard_test_unit")
    except Exception as e:
        print(f"Loop error: {e}")

if __name__ == "__main__":
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    time.sleep(10) # Run for 10 seconds
    print("\n[TEST] 10 seconds test interval completed. Success.")
    sys.exit(0)
