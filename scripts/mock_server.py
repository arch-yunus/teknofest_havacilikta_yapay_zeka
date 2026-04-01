from flask import Flask, jsonify, request, send_file
import os
import time
import random
import cv2
import numpy as np
import io

app = Flask(__name__)

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 5000
FRAME_DIR = "assets/sample_frames" # We'll create this if needed
REFERENCE_DIR = "assets/reference_objects"

# Simulation state
state = {
    "frame_index": 0,
    "total_frames": 2250,
    "gps_health": 1,
    "pos": [0.0, 0.0, 0.0],
    "video_name": "TEKNOFEST_FINAL_01",
}

# Ensure directories exist
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Helper to generate a dummy frame if none exists
def get_dummy_frame():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add some noise/texture
    cv2.putText(img, f"FRAME {state['frame_index']}", (800, 540), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    # Simulate a "landing area" as a circle
    cv2.circle(img, (1500, 800), 100, (0, 165, 255), 10) # UAP
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer)

@app.route('/api/')
def index():
    return jsonify({"status": "SkyGuard Mock Server Active", "version": "2026.1"})

@app.route('/api/frame')
def get_frame():
    """
    Simulates: http://127.0.0.1:5000/api/frame
    Returns next frame metadata + telemetry.
    """
    idx = state["frame_index"]
    
    # Simulate GPS health change after 450 frames (1 minute)
    if idx > 450:
        if random.random() < 0.2: # Intermittent health drop
            state["gps_health"] = 0
        else:
            state["gps_health"] = 1
    
    # Simulating movement
    state["pos"][0] += random.uniform(-0.5, 2.0) # Move forward/sideways
    state["pos"][1] += random.uniform(-0.5, 0.5)
    state["pos"][2] = 20.0 + random.uniform(-0.2, 0.2) # Hover at 20m

    frame_url = f"http://{HOST}:{PORT}/api/images/frame_{idx}.jpg"
    
    response = {
        "url": f"http://{HOST}:{PORT}/api/frames/{idx}",
        "image_url": frame_url,
        "video_name": state["video_name"],
        "gps_health_status": state["gps_health"],
        "translation_x": state["pos"][0] if state["gps_health"] else 0.0,
        "translation_y": state["pos"][1] if state["gps_health"] else 0.0,
        "translation_z": state["pos"][2] if state["gps_health"] else 0.0,
    }
    
    return jsonify(response)

@app.route('/api/images/<filename>')
def serve_image(filename):
    # In a real test, you'd serve actual files from FRAME_DIR
    # For mock, we just generate on the fly
    return send_file(get_dummy_frame(), mimetype='image/jpeg')

@app.route('/api/reference_objects')
def get_references():
    return jsonify([
        {"object_id": "REF_OBJ_01", "image_url": f"http://{HOST}:{PORT}/api/images/ref1.jpg"},
        {"object_id": "REF_OBJ_02", "image_url": f"http://{HOST}:{PORT}/api/images/ref2.jpg"}
    ])

@app.route('/api/results', methods=['POST'])
def post_results():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400
    
    # Validate structure (minimal check)
    required = ["user", "frame", "detected_objects", "detected_translations"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    
    print(f"[SERVER] Result Received for Frame {state['frame_index']} | {len(data['detected_objects'])} objects")
    
    # Increment frame index for next request
    state["frame_index"] += 1
    
    return jsonify({"status": "success", "message": "Result accepted"}), 201

if __name__ == '__main__':
    print(f"Starting SkyGuard Mock Server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
