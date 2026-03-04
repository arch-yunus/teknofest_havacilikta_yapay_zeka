import cv2
import numpy as np
import time
import requests
from src.vision.detector import ObjectDetector
from src.vision.tracker import CentroidTracker
from src.vision.odometry import VisualOdometry
from src.vision.matcher import ObjectMatcher
from src.telemetry.competition_client import CompetitionClient

def competition_loop():
    print("Teknofest Havacilikta Yapay Zeka - Yarismaci Yazilimi Baslatiliyor...")
    
    # Initialize components
    client = CompetitionClient(base_url="http://127.0.0.1:5000")
    detector = ObjectDetector()
    tracker = CentroidTracker()
    vo = VisualOdometry() # Task 2
    matcher = ObjectMatcher() # Task 3
    
    # Session initialization (Task 3 references)
    # In a real scenario, we'd fetch these from the server
    # matcher.add_reference_object("ref_1", cv2.imread("data/references/ref1.png"))
    
    print("Oturum Basladi. Kareler islaniyor...")
    
    try:
        while True:
            # 1. Get Frame and Telemetry
            frame_data = client.get_frame()
            if not frame_data:
                print("Bekleniyor...")
                time.sleep(1)
                continue
                
            # Download image
            img_resp = requests.get(frame_data['image_url'], stream=True)
            if img_resp.status_code == 200:
                img_bytes = np.asarray(bytearray(img_resp.raw.read()), dtype="uint8")
                frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            else:
                print("Gorsel indirilemedi.")
                continue

            # 2. Task 1: Object Detection & Tracking
            detections, _ = detector.detect(frame)
            
            # Map detections to tracker for motion status
            rects = []
            for d in detections:
                w = d['bottom_right_x'] - d['top_left_x']
                h = d['bottom_right_y'] - d['top_left_y']
                rects.append((d['top_left_x'], d['top_left_y'], w, h))
            
            tracked_objects = tracker.update(rects)
            
            # Enrich detections with motion status and object IDs
            final_detections = []
            for i, d in enumerate(detections):
                # Simple heuristic mapping for this demo
                # In real code, we'd use tracker association more rigorously
                obj_id = list(tracked_objects.keys())[i] if i < len(tracked_objects) else -1
                d['motion_status'] = tracker.get_motion_status(obj_id) if obj_id != -1 else "0"
                
                # Landing status logic (simplified: if UAİ/UAP is class 2/3)
                if d['cls'] in ["2", "3"]:
                    # Placeholder check: if any overlap with people/cars, set to 0 (Uygun Degil)
                    d['landing_status'] = "1" # Assuming suitable for now
                
                final_detections.append(d)

            # 3. Task 2: Position Estimation (Visual Odometry)
            gps_health = frame_data.get('gps_health_status', 1)
            if gps_health == 0:
                pos = vo.update(frame)
                translation = {
                    "translation_x": float(pos[0]),
                    "translation_y": float(pos[1]),
                    "translation_z": float(pos[2])
                }
            else:
                translation = {
                    "translation_x": frame_data.get('translation_x', 0.0),
                    "translation_y": frame_data.get('translation_y', 0.0),
                    "translation_z": frame_data.get('translation_z', 0.0)
                }
                vo.reset() # Align with GPS when healthy

            # 4. Task 3: Object Matching
            undefined_objects = matcher.match(frame)

            # 5. Send Results
            success = client.send_results(final_detections, translation, undefined_objects)
            
            status = "Basarili" if success else "Hata"
            print(f"Kare Islendi: {frame_data['video_name']} | Sonuc: {status}", end='\r')

            # Optional: Visual check
            # cv2.imshow("Competition Stream", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\nYarisma durduruldu.")

if __name__ == "__main__":
    competition_loop()
