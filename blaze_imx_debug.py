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
# Blaze Demo Application (live with USB camera)
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
#   plots
#      pyplotly
#      kaleido
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
# import plotly.graph_objects as go

# import matplotlib.pyplot as plt

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('blaze_common/'))
sys.path.append(os.path.abspath('blaze_tflite/'))
sys.path.append(os.path.abspath('blaze_tflite_quant/'))

from blaze_tflite.blazedetector import BlazeDetector as BlazeDetector_tflite
from blaze_tflite.blazelandmark import BlazeLandmark as BlazeLandmark_tflite
print("[INFO] blaze_tflite supported ...")

from blaze_tflite_quant.blazedetector import BlazeDetector as BlazeDetector_tflite_quant
from blaze_tflite_quant.blazelandmark import BlazeLandmark as BlazeLandmark_tflite_quant
print("[INFO] blaze_tflite_quant supported ...")

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

from timeit import default_timer as timer

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
print(' --fps         : ', args.fps)

nb_blaze_pipelines = 1
bInputImage = False
bInputVideo = False
bInputCamera = True

if os.path.exists(args.input):
    print("[INFO] Input exists : ",args.input)
    file_name, file_extension = os.path.splitext(args.input)
    file_extension = file_extension.lower()
    print("[INFO] Input type : ",file_extension)
    if file_extension == ".jpg" or file_extension == ".png" or file_extension == ".tif":
        bInputImage = True
        bInputVideo = False
        bInputCamera = False
    if file_extension == ".mov" or file_extension == ".mp4":
        bInputImage = False
        bInputVideo = True
        bInputCamera = False

if bInputCamera == True:
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

    # Open video
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;qsv|video_codec;h264_qsv|vsync;0'

    cap = cv2.VideoCapture(input_video)
    frame_width = 640
    frame_height = 480
    frame_fps = 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_fps) 
    #frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    #frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] input : camera",input_video," (",frame_width,",",frame_height,")")

if bInputVideo == True:
    # Open video file
    cap = cv2.VideoCapture(args.input)
    frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] input : video ",args.input," (",frame_width,",",frame_height,")")

if bInputImage == True:
    image = cv2.imread(args.input)
    frame_height,frame_width,_ = image.shape
    print("[INFO] input : image ",args.input," (",frame_width,",",frame_height,")")

output_dir = './captured-images'

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist

default_detector_model = "blaze_tflite/models/pose_detection_quant_floatinputs_vela.tflite"
default_landmark_model = "blaze_tflite/models/pose_landmark_full_quant_floatinputs_vela.tflite"
# default_detector_model = "blaze_tflite/models/pose_detection_128x128_integer_quant_vela.tflite" 
# default_landmark_model = "blaze_tflite/models/pose_landmark_upper_body_256x256_integer_quant_vela.tflite"
# default_detector_model = "blaze_tflite/models/output/pose_detection_quant_floatinputs_vela.tflite"
# default_landmark_model = "blaze_tflite/models/output/pose_landmark_full_quant_floatinputs_vela.tflite"
# default_detector_model = "blaze_tflite/models/output/pose_detection_128x128_integer_quant_vela.tflite"
# default_landmark_model = "blaze_tflite/models/output/pose_landmark_upper_body_256x256_integer_quant_vela.tflite"
# default_detector_model = "blaze_tflite/models/pose_detection_v0_07.tflite"
# default_landmark_model = "blaze_tflite/models/pose_landmark_v0_07_upper_body.tflite"
# default_detector_model = "blaze_tflite/models/pose_detection_quant_floatinputs.tflite"
# default_landmark_model = "blaze_tflite/models/pose_landmark_full_quant_floatinputs.tflite"
# default_detector_model = "blaze_tflite/models/pose_detection_128x128_integer_quant.tflite"
# default_landmark_model = "blaze_tflite/models/pose_landmark_upper_body_256x256_integer_quant.tflite"
# default_detector_model = "blaze_tflite/models/pose_detection.tflite"
# default_landmark_model = "blaze_tflite/models/pose_landmark_full.tflite"

blaze_detector_type = "blazepose"
blaze_landmark_type = "blazeposelandmark"
blaze_title = "BlazePoseLandmark"

if args.model1 == None:
   args.model1 = default_detector_model
if args.model2 == None:
   args.model2 = default_landmark_model

blaze_detector = BlazeDetector_tflite(blaze_detector_type)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(default_detector_model)

blaze_landmark = BlazeLandmark_tflite(blaze_landmark_type)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(default_landmark_model)


print("================================================================")
print("Blaze Detect Live Demo")
print("================================================================")

bStep = False
bPause = False
bWrite = False
bUseImage = args.image
bShowDebugImage = False
bShowScores = False
bShowFPS = args.fps
bVerbose = args.debug
bViewOutput = False
# bViewOutput = not args.withoutview

def ignore(x):
    pass

image = []
output = []

frame_count = 0

# init the real-time FPS counter
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)

while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    frame_count = frame_count + 1

    if bUseImage:
        frame = cv2.imread('woman_hands.jpg')
        if not (type(frame) is np.ndarray):
            print("[ERROR] cv2.imread('woman_hands.jpg') FAILED !")
            break;
    elif bInputImage:
        frame = cv2.imread(args.input)
        if not (type(frame) is np.ndarray):
            print("[ERROR] cv2.imread(",args.input,") FAILED !")
            break;
    else:
        flag, frame = cap.read()
        if not flag:
            print("[ERROR] cap.read() FAILEd !")
            break

    if True:    
        pipeline_id = 0
        if True:
            image = frame.copy()
                
            #image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            output2 = image.copy()
            
            # BlazePalm pipeline
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img1,scale1,pad1=blaze_detector.resize_pad(image)
            
            out1_reference,out2_reference = blaze_detector.predict_core(np.expand_dims(img1, axis=0))
            detection_boxes_reference = blaze_detector._decode_boxes(out2_reference, blaze_detector.anchors)
            #thresh = blaze_detector.score_clipping_thresh
            #clipped_score_tensor = np.clip(out1_reference,-thresh,thresh)
            #detection_scores = 1/(1 + np.exp(-clipped_score_tensor))
            detection_scores = 1/(1 + np.exp(-out1_reference))
            detection_scores_reference = np.squeeze(detection_scores, axis=-1)        
            
            normalized_detections = blaze_detector.predict_on_image(img1)
            if len(normalized_detections) > 0:
  
                detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector.detection2roi(detections)
                roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)

                flags, normalized_landmarks = blaze_landmark.predict(roi_img)

                landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if landmarks.shape[1] > 33:
                        draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                    else:
                        draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                
                   
                draw_roi(output,roi_box)
                draw_detections(output,detections)

            # display real-time FPS counter (if valid)
            if rt_fps_valid == True and bShowFPS:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)


            if False:
               if len(normalized_detections) == 0:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)]"%(
                       profile_resize+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post
                       ))
               else:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)] Extract[(%001.06f)] Landmark[(%001.06f) (%001.06f) (%001.06f)]  Annotate[(%001.06f)]"%(
                       profile_resize+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post,
                       profile_extract,                       
                       blaze_landmark.profile_pre, blaze_landmark.profile_model, blaze_landmark.profile_post,
                       profile_annotate
                       ))
            

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = 1
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        print("[INFO] ",rt_fps_message)
        rt_fps_count = 0

# Cleanup
# cv2.destroyAllWindows()
