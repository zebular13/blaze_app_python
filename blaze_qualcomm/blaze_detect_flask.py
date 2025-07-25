'''
Copyright 2024 Avnet Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import cv2
import os
from datetime import datetime
import time
import sys
import argparse
import glob
import subprocess
import re
import socket
import threading
from flask import Flask, Response, render_template_string, request, jsonify

sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

app = Flask(__name__)

# Global variables for the video stream and processing
cap = None
blaze_detector = None
blaze_landmark = None
processing_enabled = True
current_blaze_type = "hand"
frame_width = 640
frame_height = 480
output_frames = {}

def get_media_dev_by_name(src):
    devices = glob.glob("/dev/media*")
    for dev in sorted(devices):
        proc = subprocess.run(['media-ctl','-d',dev,'-p'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

def get_video_dev_by_name(src):
    devices = glob.glob("/dev/video*")
    for dev in sorted(devices):
        proc = subprocess.run(['v4l2-ctl','-d',dev,'-D'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

def initialize_camera(input_video):
    global cap, frame_width, frame_height
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap.isOpened()

def initialize_models(blaze_type, detector_model, landmark_model):
    global blaze_detector, blaze_landmark, current_blaze_type
    
    current_blaze_type = blaze_type
    
    if blaze_type == "hand":
        blaze_detector_type = "blazepalm"
        blaze_landmark_type = "blazehandlandmark"
        default_detector_model = 'models/palm_detection_lite.tflite'
        default_landmark_model = 'models/hand_landmark_lite.tflite'
    elif blaze_type == "face":
        blaze_detector_type = "blazeface"
        blaze_landmark_type = "blazefacelandmark"
        default_detector_model = 'models/face_detection_short_range.tflite'
        default_landmark_model = 'models/face_landmark.tflite'
    elif blaze_type == "pose":
        blaze_detector_type = "blazepose"
        blaze_landmark_type = "blazeposelandmark"
        default_detector_model = 'models/pose_detection.tflite'
        default_landmark_model = 'models/pose_landmark_full.tflite'
    else:
        return False

    if detector_model is None:
        detector_model = default_detector_model
    if landmark_model is None:
        landmark_model = default_landmark_model

    try:
        blaze_detector = BlazeDetector(blaze_detector_type)
        blaze_detector.load_model(detector_model)
        
        blaze_landmark = BlazeLandmark(blaze_landmark_type)
        blaze_landmark.load_model(landmark_model)
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def process_frame(frame):
    global current_blaze_type
    
    if not processing_enabled or blaze_detector is None or blaze_landmark is None:
        return frame

    image = frame.copy()
    output = image.copy()
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and pad
    img1, scale1, pad1 = blaze_detector.resize_pad(image)
    
    # Detect
    normalized_detections = blaze_detector.predict_on_image(img1)
    
    if len(normalized_detections) > 0:
        detections = blaze_detector.denormalize_detections(normalized_detections, scale1, pad1)
        xc, yc, scale, theta = blaze_detector.detection2roi(detections)
        roi_img, roi_affine, roi_box = blaze_landmark.extract_roi(image, xc, yc, theta, scale)

        predict_result = blaze_landmark.predict(roi_img)
        # print(len(predict_result))
        flags, normalized_landmarks, handedness_scores = blaze_landmark.predict(roi_img)
        landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]
            if current_blaze_type == "hand":
                draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)
            elif current_blaze_type == "face":
                draw_landmarks(output, landmark[:,:2], FACE_CONNECTIONS, size=1)
            elif current_blaze_type == "pose":
                if landmarks.shape[1] > 33:
                    draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                else:
                    draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)
        
        draw_roi(output, roi_box)
        draw_detections(output, detections)
    
    return output

def generate_frames():
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            break
            
        if processing_enabled:
            frame = process_frame(frame)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Blaze Detection Live</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { display: flex; flex-direction: column; align-items: center; }
                .controls { margin: 20px 0; }
                button { padding: 8px 16px; margin: 0 5px; cursor: pointer; }
                select { padding: 8px; margin: 0 5px; }
                .video-container { position: relative; }
                .fps-counter { 
                    position: absolute; 
                    top: 10px; 
                    left: 10px; 
                    background: rgba(0,0,0,0.7); 
                    color: white; 
                    padding: 5px 10px; 
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Blaze Detection Live</h1>
                
                <div class="controls">
                    <button onclick="toggleProcessing()">Toggle Processing</button>
                    <select id="modelType" onchange="changeModelType()">
                        <option value="hand">Hand Detection</option>
                        <option value="face">Face Detection</option>
                        <option value="pose">Pose Detection</option>
                    </select>
                    <button onclick="captureFrame()">Capture Frame</button>
                </div>
                
                <div class="video-container">
                    <div class="fps-counter" id="fpsCounter">FPS: --</div>
                    <img src="/video_feed" width="640" height="480">
                </div>
            </div>
            
            <script>
                let fps = 0;
                let frameCount = 0;
                let lastTime = performance.now();
                
                // Update FPS counter
                function updateFPS() {
                    frameCount++;
                    const now = performance.now();
                    if (now - lastTime >= 1000) {
                        fps = frameCount;
                        frameCount = 0;
                        lastTime = now;
                        document.getElementById('fpsCounter').innerText = `FPS: ${fps}`;
                    }
                    requestAnimationFrame(updateFPS);
                }
                updateFPS();
                
                function toggleProcessing() {
                    fetch('/toggle_processing')
                        .then(response => response.json())
                        .then(data => {
                            console.log('Processing toggled:', data.processing_enabled);
                        });
                }
                
                function changeModelType() {
                    const modelType = document.getElementById('modelType').value;
                    fetch('/change_model/' + modelType)
                        .then(response => response.json())
                        .then(data => {
                            console.log('Model changed:', data.message);
                        });
                }
                
                function captureFrame() {
                    fetch('/capture_frame')
                        .then(response => response.json())
                        .then(data => {
                            console.log('Frame captured:', data.message);
                            alert('Frame captured successfully!');
                        });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing')
def toggle_processing():
    global processing_enabled
    processing_enabled = not processing_enabled
    return jsonify({'processing_enabled': processing_enabled})

@app.route('/change_model/<model_type>')
def change_model(model_type):
    if model_type in ['hand', 'face', 'pose']:
        success = initialize_models(model_type, None, None)
        if success:
            return jsonify({'message': f'Model changed to {model_type}'})
        else:
            return jsonify({'error': 'Failed to change model'}), 500
    else:
        return jsonify({'error': 'Invalid model type'}), 400

@app.route('/capture_frame')
def capture_frame():
    if cap is None or not cap.isOpened():
        return jsonify({'error': 'Camera not available'}), 400
        
    success, frame = cap.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'}), 500
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_frame_{timestamp}.jpg"
    
    if not os.path.exists('captured-images'):
        os.makedirs('captured-images')
    
    try:
        cv2.imwrite(f'captured-images/{filename}', frame)
        return jsonify({'message': f'Frame saved as {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="", 
                       help="Video input device. Default is auto-detect (first usbcam)")
    parser.add_argument('-b', '--blaze', type=str, default="hand", 
                       help="Application (hand, face, pose). Default is hand")
    parser.add_argument('-m', '--model1', type=str, 
                       help='Path of blazepalm model. Default is models/palm_detection_lite.tflite')
    parser.add_argument('-n', '--model2', type=str, 
                       help='Path of blazehandlandmark model. Default is models/hand_landmark_lite.tflite')
    args = parser.parse_args()

    # Initialize camera
    print("[INFO] Searching for USB camera ...")
    dev_video = get_video_dev_by_name("uvcvideo")
    dev_media = get_media_dev_by_name("uvcvideo")
    print(dev_video)
    print(dev_media)

    if dev_video is None:
        input_video = 0
    elif args.input != "":
        input_video = args.input 
    else:
        input_video = dev_video  
    print("[INFO] Input Video : ", input_video)

    if not initialize_camera(input_video):
        print("[ERROR] Failed to initialize camera")
        exit(1)

    # Initialize models
    if not initialize_models(args.blaze, args.model1, args.model2):
        print("[ERROR] Failed to initialize models")
        exit(1)

    # Start Flask app
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, threaded=True)