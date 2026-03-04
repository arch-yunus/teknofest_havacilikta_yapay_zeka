import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.detector import ObjectDetector
from src.vision.tracker import CentroidTracker

class TestCompetitionVision(unittest.TestCase):
    def test_class_mapping(self):
        detector = ObjectDetector()
        self.assertEqual(detector.class_map.get('person'), 1)
        self.assertEqual(detector.class_map.get('bus'), 0)
        self.assertEqual(detector.class_map.get('parking'), 2)

    def test_motion_status(self):
        tracker = CentroidTracker()
        # Register an object
        tracker.register(np.array([100, 100]))
        obj_id = 0
        
        # Simulate movement
        tracker.kalmans[obj_id].state[2:] = [5.0, 5.0] # High velocity
        self.assertEqual(tracker.get_motion_status(obj_id), "1")
        
        # Simulate static
        tracker.kalmans[obj_id].state[2:] = [0.1, 0.1] # Low velocity
        self.assertEqual(tracker.get_motion_status(obj_id), "0")

if __name__ == '__main__':
    unittest.main()
