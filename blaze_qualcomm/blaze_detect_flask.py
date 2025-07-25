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
#
# Blaze Demo Application (Flask Web Version)
#
# References:
#   https://www.github.com/AlbertaBeef/blaze_app_python
#   https://www.github.com/AlbertaBeef/blaze_tutorial/tree/2023.1
#
# Dependencies:
#   TFLite
#      tensorflow
#    or
#      tflite_runtime
#   Web
#      flask
#      plotly
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

from ctypes import *
from typing import List
import pathlib
#import threading
import time
import sys
import argparse
import glob
import subprocess
import re
import sys

from datetime import datetime
import plotly.graph_objects as go

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

from timeit import default_timer as timer
from flask import Flask, Response, render_template_string, request, jsonify

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


# Parameters (tweaked for video)
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input'      , type=str, default="", help="Video input device. Default is auto-detect (first usbcam)")
ap.add_argument('-I', '--image'      , default=False, action='store_true', help="Use 'womand_hands.jpg' image as input. Default is usbcam")
ap.add_argument('-b', '--blaze',  type=str, default="hand", help="Application (hand, face, pose).  Default is hand")
ap.add_argument('-m', '--model1', type=str, help='Path of blazepalm model. Default is models/palm_detection_without_custom_op.tflite')
ap.add_argument('-n', '--model2', type=str, help='Path of blazehandlardmark model. Default is models/hand_landmark.tflite')
ap.add_argument('-d', '--debug'      , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-z', '--profilelog' , default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
ap.add_argument('-Z', '--profileview', default=False, action='store_true', help="Enable Profile View (Latency). Default is off")
ap.add_argument('-f', '--fps'        , default=False, action='store_true', help="Enable FPS display. Default is off")

args = ap.parse_args()  
  
print('Command line options:')
print(' --input       : ', args.input)
print(' --image       : ', args.image)
print(' --blaze       : ', args.blaze)
print(' --model1      : ', args.model1)
print(' --model2      : ', args.model2)
print(' --debug       : ', args.debug)
print(' --withoutview : ', args.withoutview)
print(' --profilelog  : ', args.profilelog)
print(' --profileview  : ', args.profileview)
print(' --fps         : ', args.fps)

nb_blaze_pipelines = 1

print("[INFO] Searching for USB camera ...")
dev_video = get_video_dev_by_name("uvcvideo")
dev_media = get_media_dev_by_name("uvcvideo")
print(dev_video)
print(dev_media)

if dev_video == None:
    input_video = 0
elif args.input != "":
    input_video = args.input 
else:
    input_video = dev_video  
print("[INFO] Input Video : ",input_video)

output_dir = './captured-images'

profile_csv = './blaze_detect_live.csv'
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :",profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :",profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist

# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("camera",input_video," (",frame_width,",",frame_height,")")


if args.blaze == "hand":
   blaze_detector_type = "blazepalm"
   blaze_landmark_type = "blazehandlandmark"
   blaze_title = "BlazeHandLandmark"
   default_detector_model='models/palm_detection_lite.tflite'
   default_landmark_model='models/hand_landmark_lite.tflite'
elif args.blaze == "face":
   blaze_detector_type = "blazeface"
   blaze_landmark_type = "blazefacelandmark"
   blaze_title = "BlazeFaceLandmark"
   default_detector_model='models/face_detection_short_range.tflite'
   default_landmark_model='models/face_landmark.tflite'
elif args.blaze == "pose":
   blaze_detector_type = "blazepose"
   blaze_landmark_type = "blazeposelandmark"
   blaze_title = "BlazePoseLandmark"
   default_detector_model='models/pose_detection.tflite'
   default_landmark_model='models/pose_landmark_full.tflite'
else:
   print("[ERROR] Invalid Blaze application : ",args.blaze,".  MUST be one of hand,face,pose.")

if args.model1 == None:
   args.model1 = default_detector_model
if args.model2 == None:
   args.model2 = default_landmark_model

blaze_detector = BlazeDetector(blaze_detector_type)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(args.model1)

blaze_landmark = BlazeLandmark(blaze_landmark_type)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(args.model2)


print("================================================================")
print("Blaze Detect Live Demo (Flask Web Version)")
print("================================================================")
print("\tAccess the web interface at http://localhost:5000")
print("----------------------------------------------------------------")
print("\tPress CTRL+C to quit ...")
print("================================================================")

# Flask app setup
app = Flask(__name__)

# Global state variables
bStep = False
bPause = False
bWrite = False
bUseImage = args.image
bShowDebugImage = False
bShowScores = False
bShowFPS = args.fps
bVerbose = args.debug
bViewOutput = not args.withoutview
bProfileLog = args.profilelog
bProfileView = args.profileview

# Initialize profile images
last_profile_img = None
last_fps_img = None

def generate_frames():
    global bStep, bPause, bWrite, bUseImage, bShowDebugImage, bShowScores
    global bShowFPS, bVerbose, bViewOutput, bProfileLog, bProfileView
    global last_profile_img, last_fps_img
    
    frame_count = 0
    rt_fps_count = 0
    rt_fps_time = cv2.getTickCount()
    rt_fps_valid = False
    rt_fps = 0.0
    
    while True:
        if bPause and not bStep:
            time.sleep(0.1)
            continue
            
        bStep = False
        
        # FPS counter
        if rt_fps_count == 0:
            rt_fps_time = cv2.getTickCount()

        frame_count = frame_count + 1

        if bUseImage:
            frame = cv2.imread('../woman_hands.jpg')
            if not (type(frame) is np.ndarray):
                print("[ERROR] cv2.imread('woman_hands.jpg') FAILED !")
                break;
        else:
            flag, frame = cap.read()
            if not flag:
                print("[ERROR] cap.read() FAILED !")
                break

        if bProfileLog or bProfileView:
            prof_title          = ['']*nb_blaze_pipelines
            prof_resize         = np.zeros(nb_blaze_pipelines)
            prof_detector_pre   = np.zeros(nb_blaze_pipelines)
            prof_detector_model = np.zeros(nb_blaze_pipelines)
            prof_detector_post  = np.zeros(nb_blaze_pipelines)
            prof_extract_roi    = np.zeros(nb_blaze_pipelines)
            prof_landmark_pre   = np.zeros(nb_blaze_pipelines)
            prof_landmark_model = np.zeros(nb_blaze_pipelines)
            prof_landmark_post  = np.zeros(nb_blaze_pipelines)
            prof_annotate       = np.zeros(nb_blaze_pipelines)
            prof_total          = np.zeros(nb_blaze_pipelines)
            prof_fps            = np.zeros(nb_blaze_pipelines)

        if True:    
            pipeline_id = 0
            if True:
                image = frame.copy()
                output = image.copy()
                
                # BlazePalm pipeline
                start = timer()
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                img1,scale1,pad1=blaze_detector.resize_pad(image)
                profile_resize = timer()-start

                if bShowDebugImage:
                    debug_img = img1.astype(np.float32)/255.0
                    debug_img = cv2.resize(debug_img,(blaze_landmark.resolution,blaze_landmark.resolution))
                
                normalized_detections = blaze_detector.predict_on_image(img1)
                if len(normalized_detections) > 0:
                    start = timer()          
                    detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    xc,yc,scale,theta = blaze_detector.detection2roi(detections)
                    roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
                    profile_extract = timer()-start

                    # Handle different return types based on model
                    predict_result = blaze_landmark.predict(roi_img)
                    if blaze_landmark_type == "blazehandlandmark":
                        flags, normalized_landmarks, handedness_scores = predict_result
                    else:
                        flags, normalized_landmarks = predict_result

                    if bShowDebugImage:
                        for i in range(roi_img.shape[0]):
                            roi_landmarks = normalized_landmarks[i,:,:].copy()
                            roi_landmarks = roi_landmarks*blaze_landmark.resolution
                            if blaze_landmark_type == "blazehandlandmark":
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], HAND_CONNECTIONS, size=2)
                            elif blaze_landmark_type == "blazefacelandmark":
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], FACE_CONNECTIONS, size=1)                                    
                            elif blaze_landmark_type == "blazeposelandmark":
                                if roi_landmarks.shape[1] > 33:
                                    draw_landmarks(roi_img[i], roi_landmarks[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                                else:
                                    draw_landmarks(roi_img[i], roi_landmarks[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                
                            debug_img = cv2.hconcat([debug_img,roi_img[i]])

                    start = timer() 
                    landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

                    for i in range(len(flags)):
                        landmark, flag = landmarks[i], flags[i]
                        if blaze_landmark_type == "blazehandlandmark":
                            draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)
                        elif blaze_landmark_type == "blazefacelandmark":
                            draw_landmarks(output, landmark[:,:2], FACE_CONNECTIONS, size=1)                                    
                        elif blaze_landmark_type == "blazeposelandmark":
                            if landmarks.shape[1] > 33:
                                draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                            else:
                                draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                
                       
                    draw_roi(output,roi_box)
                    draw_detections(output,detections)
                    profile_annotate = timer()-start

                if bShowDebugImage:
                    if debug_img.shape[0] == debug_img.shape[1]:
                        zero_img = np.full_like(debug_img,0.0)
                        debug_img = cv2.hconcat([debug_img,zero_img])
                    debug_img = cv2.cvtColor(debug_img,cv2.COLOR_RGB2BGR)

                # FPS display
                if rt_fps_valid == True and bShowFPS:
                    rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
                    rt_fps_x = int(10*scale)
                    rt_fps_y = int((frame_height-10)*scale)
                    cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)

                # Profiling
                if bProfileLog or bProfileView:
                   prof_title[pipeline_id] = blaze_title
                   prof_resize[pipeline_id]         = profile_resize
                   prof_detector_pre[pipeline_id]   = blaze_detector.profile_pre
                   prof_detector_model[pipeline_id] = blaze_detector.profile_model
                   prof_detector_post[pipeline_id]  = blaze_detector.profile_post
                   if len(normalized_detections) > 0:
                       prof_extract_roi[pipeline_id]    = profile_extract
                       prof_landmark_pre[pipeline_id]   = blaze_landmark.profile_pre
                       prof_landmark_model[pipeline_id] = blaze_landmark.profile_model
                       prof_landmark_post[pipeline_id]  = blaze_landmark.profile_post
                       prof_annotate[pipeline_id]       = profile_annotate
                   #
                   prof_total[pipeline_id] = profile_resize + \
                                             blaze_detector.profile_pre + \
                                             blaze_detector.profile_model + \
                                             blaze_detector.profile_post
                   if len(normalized_detections) > 0:
                       prof_total[pipeline_id] += profile_extract + \
                                                  blaze_landmark.profile_pre + \
                                                  blaze_landmark.profile_model + \
                                                  blaze_landmark.profile_post + \
                                                  profile_annotate
                   prof_fps[pipeline_id] = 1.0 / prof_total[pipeline_id]

                   # Generate profile visualization
                   if bProfileView:
                       # Latency visualization
                       fig_latency = go.Figure(data=[
                           go.Bar(name='resize', y=['Pipeline'], x=[prof_resize[pipeline_id]], orientation='h'),
                           go.Bar(name='detector[pre]', y=['Pipeline'], x=[prof_detector_pre[pipeline_id]], orientation='h'),
                           go.Bar(name='detector[model]', y=['Pipeline'], x=[prof_detector_model[pipeline_id]], orientation='h'),
                           go.Bar(name='detector[post]', y=['Pipeline'], x=[prof_detector_post[pipeline_id]], orientation='h'),
                           go.Bar(name='extract_roi', y=['Pipeline'], x=[prof_extract_roi[pipeline_id]], orientation='h'),
                           go.Bar(name='landmark[pre]', y=['Pipeline'], x=[prof_landmark_pre[pipeline_id]], orientation='h'),
                           go.Bar(name='landmark[model]', y=['Pipeline'], x=[prof_landmark_model[pipeline_id]], orientation='h'),
                           go.Bar(name='landmark[post]', y=['Pipeline'], x=[prof_landmark_post[pipeline_id]], orientation='h'),
                           go.Bar(name='annotate', y=['Pipeline'], x=[prof_annotate[pipeline_id]], orientation='h')
                       ])
                       fig_latency.update_layout(title='Latency (sec)', barmode='stack')
                       last_profile_img = fig_latency.to_image(format="png")

                       # FPS visualization
                       fig_fps = go.Figure(data=[
                           go.Bar(name='FPS', y=['Pipeline'], x=[prof_fps[pipeline_id]], orientation='h')
                       ])
                       fig_fps.update_layout(title='Performance (FPS)')
                       last_fps_img = fig_fps.to_image(format="png")

                # Write to CSV if enabled
                if bProfileLog:            
                    timestamp = datetime.now()
                    pipeline_id = 0
                    if True:
                            csv_str = \
                                str(timestamp)+","+\
                                str(user)+","+\
                                str(host)+","+\
                                "blaze_tflite"+","+\
                                str(prof_resize[pipeline_id])+","+\
                                str(prof_detector_pre[pipeline_id])+","+\
                                str(prof_detector_model[pipeline_id])+","+\
                                str(prof_detector_post[pipeline_id])+","+\
                                str(prof_extract_roi[pipeline_id])+","+\
                                str(prof_landmark_pre[pipeline_id])+","+\
                                str(prof_landmark_model[pipeline_id])+","+\
                                str(prof_landmark_post[pipeline_id])+","+\
                                str(prof_annotate[pipeline_id])+","+\
                                str(prof_total[pipeline_id])+","+\
                                str(prof_fps[pipeline_id])+"\n"
                            f_profile_csv.write(csv_str)

                # Capture frame if requested
                if bWrite:
                    filename = ("blaze_detect_live_frame%04d_%s_input.tif"%(frame_count,blaze_title))
                    print("Capturing ",filename," ...")
                    input_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir,filename),input_img)

                    filename = ("blaze_detect_live_frame%04d_%s_detection.tif"%(frame_count,blaze_title))
                    print("Capturing ",filename," ...")
                    cv2.imwrite(os.path.join(output_dir,filename),output)
            
                    if bShowDebugImage:
                        filename = ("blaze_detect_live_frame%04d_%s_debug.tif"%(frame_count,blaze_title))
                        print("Capturing ",filename," ...")
                        cv2.imwrite(os.path.join(output_dir,filename),debug_img)

        # Update FPS counter
        rt_fps_count = rt_fps_count + 1
        if rt_fps_count == 10:
            t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
            rt_fps_valid = 1
            rt_fps = 10.0/t
            rt_fps_count = 0

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', output)
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing')
def toggle_processing():
    global processing_enabled
    processing_enabled = not processing_enabled
    return jsonify({'processing_enabled': processing_enabled})

@app.route('/change_model/<model_type>')
def change_model(model_type):
    global blaze_detector, blaze_landmark, current_blaze_type
    if model_type in ['hand', 'face', 'pose']:
        if model_type == "hand":
            blaze_detector_type = "blazepalm"
            blaze_landmark_type = "blazehandlandmark"
            default_detector_model='models/palm_detection_lite.tflite'
            default_landmark_model='models/hand_landmark_lite.tflite'
        elif model_type == "face":
            blaze_detector_type = "blazeface"
            blaze_landmark_type = "blazefacelandmark"
            default_detector_model='models/face_detection_short_range.tflite'
            default_landmark_model='models/face_landmark.tflite'
        elif model_type == "pose":
            blaze_detector_type = "blazepose"
            blaze_landmark_type = "blazeposelandmark"
            default_detector_model='models/pose_detection.tflite'
            default_landmark_model='models/pose_landmark_full.tflite'
        
        try:
            blaze_detector = BlazeDetector(blaze_detector_type)
            blaze_detector.load_model(default_detector_model)
            blaze_landmark = BlazeLandmark(blaze_landmark_type)
            blaze_landmark.load_model(default_landmark_model)
            current_blaze_type = model_type
            return jsonify({'message': f'Model changed to {model_type}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid model type'}), 400

@app.route('/capture_frame')
def capture_frame():
    global bWrite
    bWrite = True
    return jsonify({'message': 'Frame capture triggered'})

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)

    # Cleanup when Flask app stops
    f_profile_csv.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()