import requests
import json
import time

class CompetitionClient:
    def __init__(self, base_url="http://127.0.0.1:5000", username="team_skyguard"):
        self.base_url = base_url
        self.username = username
        self.session_url = None
        self.frame_data = None

    def get_frame(self):
        """
        Fetches the next video frame and associated telemetry from the competition server.
        As per spec Chapter 8.
        """
        try:
            response = requests.get(f"{self.base_url}/api/frame")
            if response.status_code == 200:
                self.frame_data = response.json()
                return self.frame_data
            else:
                print(f"Error fetching frame: {response.status_code}")
                return None
        except Exception as e:
            print(f"Connection error: {e}")
            return None

    def send_results(self, detections, translation, undefined_objects=None):
        """
        Sends the detection and position results back to the server.
        Format based on Figure 17 of the technical specification.
        """
        if not self.frame_data:
            print("No frame data to associate results with.")
            return False

        payload = {
            "user": f"{self.base_url}/api/users/{self.username}/",
            "frame": self.frame_data.get("url"),
            "detected_objects": detections,
            "detected_translations": [translation],
            "detected_undefined_objects": undefined_objects or []
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/results",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in [200, 201]:
                return True
            else:
                print(f"Error sending results: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Connection error while sending results: {e}")
            return False

if __name__ == "__main__":
    # Simple test/mock usage
    client = CompetitionClient()
    frame = client.get_frame()
    if frame:
        print(f"Fetched frame: {frame.get('video_name')} - {frame.get('image_url')}")
