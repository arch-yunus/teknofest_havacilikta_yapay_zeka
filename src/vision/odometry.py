import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, camera_matrix=None):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.prev_frame = None
        self.prev_kps = None
        self.prev_des = None
        
        # Identity or provided camera matrix
        if camera_matrix is None:
            self.K = np.array([[1000, 0, 960],
                               [0, 1000, 540],
                               [0, 0, 1]], dtype=np.float32)
        else:
            self.K = camera_matrix

        self.current_pos = np.zeros(3) # [x, y, z] relative to start
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def update(self, frame, dt=0.133): # ~7.5 FPS
        """
        Estimates displacement between current and previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_des = des
            return self.current_pos

        # Match features
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extraction of matching points
        pts1 = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

        if len(matches) > 10:
            # Estimate Essential Matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
                
                # Simplified scaling (Task 2 provides altitude, which we could use to scale t)
                # For now, we assume unit scale or based on dt
                self.current_pos += t.flatten() * 0.5 # Dummy scaling factor

        self.prev_frame = gray
        self.prev_kps = kps
        self.prev_des = des

        return self.current_pos

    def reset(self):
        self.current_pos = np.zeros(3)
        self.prev_frame = None
