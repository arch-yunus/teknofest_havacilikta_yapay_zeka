import numpy as np
from collections import OrderedDict

class KalmanFilter:
    def __init__(self, dt=0.1, process_variance=1e-5, measurement_variance=1e-1):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1.0 # State covariance
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_variance
        self.R = np.eye(2) * measurement_variance

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict() # ID -> Centroid
        self.disappeared = OrderedDict() # ID -> Disappeared count
        self.kalmans = OrderedDict() # ID -> KalmanFilter
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        kf = KalmanFilter()
        kf.state[:2] = centroid
        self.kalmans[self.next_object_id] = kf
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.kalmans[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Predict even if not seen
                self.objects[object_id] = self.kalmans[object_id].predict().astype(int)
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            
            # Predict for all
            predictions = []
            for oid in object_ids:
                predictions.append(self.kalmans[oid].predict())
            
            D = np.zeros((len(predictions), len(input_centroids)))
            for i in range(len(predictions)):
                for j in range(len(input_centroids)):
                    dist = np.linalg.norm(predictions[i] - input_centroids[j])
                    D[i, j] = dist
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Distance threshold for matching
                if D[row, col] > 100: # Threshold in pixels
                    continue

                object_id = object_ids[row]
                self.kalmans[object_id].update(input_centroids[col])
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            # Unused rows (disappeared)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                self.objects[object_id] = self.kalmans[object_id].predict().astype(int)
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Unused cols (new)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    def get_motion_status(self, object_id):
        """
        Returns motion status based on velocity from Kalman Filter.
        0: Hareketsiz, 1: Hareketli
        """
        if object_id not in self.kalmans:
            return "0"
        
        # State: [x, y, vx, vy]
        vx, vy = self.kalmans[object_id].state[2:]
        velocity = np.sqrt(vx**2 + vy**2)
        
        # Threshold for camera motion vs actual motion (simplified)
        return "1" if velocity > 2.0 else "0"
