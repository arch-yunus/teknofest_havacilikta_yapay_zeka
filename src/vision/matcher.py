import cv2
import numpy as np

class ObjectMatcher:
    def __init__(self):
        # Using SIFT for better scale/rotation invariance as required by Task 3
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.reference_objects = {} # ID -> (kps, des, img_shape)

    def add_reference_object(self, object_id, image):
        """
        Extracts features from a reference image provided at session start.
        """
        if image is None: return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self.sift.detectAndCompute(gray, None)
        self.reference_objects[object_id] = (kps, des, gray.shape)

    def match(self, frame):
        """
        Searches for all registered reference objects in the given frame.
        Returns a list of detected reference objects with bounding boxes.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        matches_found = []

        if des_frame is None: return matches_found

        for obj_id, (kps_ref, des_ref, shape_ref) in self.reference_objects.items():
            # Match features
            matches = self.bf.knnMatch(des_ref, des_frame, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                # Find homography
                src_pts = np.float32([kps_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = shape_ref
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Calculate bounding box
                    x, y, w_box, h_box = cv2.boundingRect(dst)
                    
                    matches_found.append({
                        "object_id": obj_id,
                        "top_left_x": int(x),
                        "top_left_y": int(y),
                        "bottom_right_x": int(x + w_box),
                        "bottom_right_y": int(y + h_box)
                    })

        return matches_found
