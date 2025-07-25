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
import plotly.graph_objects as go
from io import BytesIO

sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

app = Flask(__name__)

# Global variables
cap = None
blaze_detector = None
blaze_landmark = None
processing_enabled = True
current_blaze_type = "hand"
frame_width = 640
frame_height = 480
output_frames = {}
profile_data = {
    'enable_log': False,
    'enable_view': False,
    'csv_file': './blaze_detect_live.csv',
    'last_profile_img': None,
    'last_fps_img': None
}

# Initialize directories
if not os.path.exists('captured-images'):
    os.makedirs('captured-images')

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

def process_frame(frame, frame_count):
    global current_blaze_type, profile_data
    
    if not processing_enabled or blaze_detector is None or blaze_landmark is None:
        return frame, None, None

    # Initialize profiling variables
    if profile_data['enable_log'] or profile_data['enable_view']:
        prof_resize = 0
        prof_detector_pre = 0
        prof_detector_model = 0
        prof_detector_post = 0
        prof_extract_roi = 0
        prof_landmark_pre = 0
        prof_landmark_model = 0
        prof_landmark_post = 0
        prof_annotate = 0
        prof_total = 0
        prof_fps = 0

    start_total = time.time()
    
    image = frame.copy()
    output = image.copy()
    
    # Convert to RGB
    start = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    profile_resize = time.time() - start
    
    # Resize and pad
    start = time.time()
    img1, scale1, pad1 = blaze_detector.resize_pad(image)
    profile_resize += time.time() - start
    
    # Detect
    normalized_detections = blaze_detector.predict_on_image(img1)
    
    if len(normalized_detections) > 0:
        start = time.time()          
        detections = blaze_detector.denormalize_detections(normalized_detections, scale1, pad1)
        xc, yc, scale, theta = blaze_detector.detection2roi(detections)
        roi_img, roi_affine, roi_box = blaze_landmark.extract_roi(image, xc, yc, theta, scale)
        profile_extract = time.time() - start

        # Handle different return types based on model
        predict_result = blaze_landmark.predict(roi_img)
        
        if current_blaze_type == "hand":
            flags, normalized_landmarks, handedness_scores = predict_result
        else:
            flags, normalized_landmarks = predict_result

        start = time.time() 
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
        profile_annotate = time.time() - start

    # Calculate profiling data
    if profile_data['enable_log'] or profile_data['enable_view']:
        prof_resize = profile_resize
        prof_detector_pre = blaze_detector.profile_pre
        prof_detector_model = blaze_detector.profile_model
        prof_detector_post = blaze_detector.profile_post
        if len(normalized_detections) > 0:
            prof_extract_roi = profile_extract
            prof_landmark_pre = blaze_landmark.profile_pre
            prof_landmark_model = blaze_landmark.profile_model
            prof_landmark_post = blaze_landmark.profile_post
            prof_annotate = profile_annotate
        
        prof_total = prof_resize + prof_detector_pre + prof_detector_model + prof_detector_post
        if len(normalized_detections) > 0:
            prof_total += prof_extract_roi + prof_landmark_pre + prof_landmark_model + prof_landmark_post + prof_annotate
        prof_fps = 1.0 / prof_total if prof_total > 0 else 0

        # Write to CSV if enabled
        if profile_data['enable_log']:
            timestamp = datetime.now()
            user = os.getenv('USER', 'unknown')
            host = socket.gethostname()
            
            csv_str = (
                f"{timestamp},{user},{host},blaze_tflite,"
                f"{prof_resize},{prof_detector_pre},{prof_detector_model},{prof_detector_post},"
                f"{prof_extract_roi},{prof_landmark_pre},{prof_landmark_model},{prof_landmark_post},"
                f"{prof_annotate},{prof_total},{prof_fps}\n"
            )
            
            with open(profile_data['csv_file'], 'a') as f:
                f.write(csv_str)

        # Generate profile visualization if enabled
        if profile_data['enable_view']:
            # Latency visualization
            fig_latency = go.Figure(data=[
                go.Bar(name='resize', y=['Pipeline'], x=[prof_resize], orientation='h'),
                go.Bar(name='detector[pre]', y=['Pipeline'], x=[prof_detector_pre], orientation='h'),
                go.Bar(name='detector[model]', y=['Pipeline'], x=[prof_detector_model], orientation='h'),
                go.Bar(name='detector[post]', y=['Pipeline'], x=[prof_detector_post], orientation='h'),
                go.Bar(name='extract_roi', y=['Pipeline'], x=[prof_extract_roi], orientation='h'),
                go.Bar(name='landmark[pre]', y=['Pipeline'], x=[prof_landmark_pre], orientation='h'),
                go.Bar(name='landmark[model]', y=['Pipeline'], x=[prof_landmark_model], orientation='h'),
                go.Bar(name='landmark[post]', y=['Pipeline'], x=[prof_landmark_post], orientation='h'),
                go.Bar(name='annotate', y=['Pipeline'], x=[prof_annotate], orientation='h')
            ])
            
            fig_latency.update_layout(
                title='Latency (sec)',
                xaxis_title='Latency',
                yaxis_title='Pipeline',
                legend_title="Component:",
                barmode='stack'
            )
            
            # Convert to image
            img_bytes = fig_latency.to_image(format="png")
            profile_latency_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            profile_data['last_profile_img'] = profile_latency_img

            # FPS visualization
            fig_fps = go.Figure(data=[
                go.Bar(name='FPS', y=['Pipeline'], x=[prof_fps], orientation='h')
            ])
            
            fig_fps.update_layout(
                title='Performance (FPS)',
                xaxis_title='FPS',
                yaxis_title='Pipeline'
            )
            
            # Convert to image
            img_bytes = fig_fps.to_image(format="png")
            profile_fps_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            profile_data['last_fps_img'] = profile_fps_img

    return output, profile_data['last_profile_img'], profile_data['last_fps_img']

def generate_frames():
    frame_count = 0
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        if processing_enabled:
            frame, profile_img, fps_img = process_frame(frame, frame_count)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
                .controls { margin: 20px 0; display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
                button { padding: 8px 16px; cursor: pointer; }
                select { padding: 8px; }
                .video-container { position: relative; margin-bottom: 20px; }
                .fps-counter { 
                    position: absolute; 
                    top: 10px; 
                    left: 10px; 
                    background: rgba(0,0,0,0.7); 
                    color: white; 
                    padding: 5px 10px; 
                    border-radius: 5px;
                }
                .profile-container { 
                    display: flex; 
                    justify-content: space-around; 
                    width: 100%; 
                    margin-top: 20px;
                }
                .profile-box { 
                    border: 1px solid #ddd; 
                    padding: 10px; 
                    margin: 10px; 
                    text-align: center;
                }
                .profile-img { max-width: 100%; }
                .row { display: flex; width: 100%; }
                .col { flex: 1; padding: 10px; }
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
                    <button onclick="toggleProfileLog()">Toggle Profile Log</button>
                    <button onclick="toggleProfileView()">Toggle Profile View</button>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="video-container">
                            <div class="fps-counter" id="fpsCounter">FPS: --</div>
                            <img src="/video_feed" width="640" height="480">
                        </div>
                    </div>
                    
                    <div class="col" id="profileView" style="display: none;">
                        <h2>Performance Metrics</h2>
                        <div class="profile-container">
                            <div class="profile-box">
                                <h3>Latency Breakdown</h3>
                                <img src="/profile_feed" class="profile-img">
                            </div>
                            <div class="profile-box">
                                <h3>FPS</h3>
                                <img src="/fps_feed" class="profile-img">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let fps = 0;
                let frameCount = 0;
                let lastTime = performance.now();
                
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
                
                function toggleProfileLog() {
                    fetch('/toggle_profile_log')
                        .then(response => response.json())
                        .then(data => {
                            console.log('Profile log toggled:', data.enabled);
                        });
                }
                
                function toggleProfileView() {
                    const profileView = document.getElementById('profileView");
                    if (profileView.style.display === "none") {
                        profileView.style.display = "block";
                    } else {
                        profileView.style.display = "none";
                    }
                    fetch('/toggle_profile_view')
                        .then(response => response.json())
                        .then(data => {
                            console.log('Profile view toggled:', data.enabled);
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

@app.route('/profile_feed')
def profile_feed():
    def generate():
        while True:
            if profile_data['last_profile_img'] is not None:
                ret, buffer = cv2.imencode('.jpg', profile_data['last_profile_img'])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fps_feed')
def fps_feed():
    def generate():
        while True:
            if profile_data['last_fps_img'] is not None:
                ret, buffer = cv2.imencode('.jpg', profile_data['last_fps_img'])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing')
def toggle_processing():
    global processing_enabled
    processing_enabled = not processing_enabled
    return jsonify({'processing_enabled': processing_enabled})

@app.route('/change_model/<model_type>')
def change_model(model_type):
    global current_blaze_type
    if model_type in ['hand', 'face', 'pose']:
        success = initialize_models(model_type, None, None)
        if success:
            current_blaze_type = model_type
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
    
    try:
        cv2.imwrite(f'captured-images/{filename}', frame)
        return jsonify({'message': f'Frame saved as {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_profile_log')
def toggle_profile_log():
    global profile_data
    profile_data['enable_log'] = not profile_data['enable_log']
    
    # Initialize CSV file if enabling
    if profile_data['enable_log'] and not os.path.isfile(profile_data['csv_file']):
        with open(profile_data['csv_file'], 'w') as f:
            f.write("time,user,hostname,pipeline,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")
    
    return jsonify({'enabled': profile_data['enable_log']})

@app.route('/toggle_profile_view')
def toggle_profile_view():
    global profile_data
    profile_data['enable_view'] = not profile_data['enable_view']
    return jsonify({'enabled': profile_data['enable_view']})

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
    parser.add_argument('-z', '--profilelog', action='store_true', 
                       help='Enable profile logging')
    parser.add_argument('-Z', '--profileview', action='store_true', 
                       help='Enable profile visualization')
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

    # Set initial profile states
    profile_data['enable_log'] = args.profilelog
    profile_data['enable_view'] = args.profileview

    # Initialize CSV file if logging is enabled
    if profile_data['enable_log'] and not os.path.isfile(profile_data['csv_file']):
        with open(profile_data['csv_file'], 'w') as f:
            f.write("time,user,hostname,pipeline,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")

    # Start Flask app
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, threaded=True)