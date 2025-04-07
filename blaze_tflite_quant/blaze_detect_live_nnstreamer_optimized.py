'''
Optimized BlazePose with NNStreamer - Full Implementation with Profiling
Copyright 2024 Avnet Inc.
Licensed under the Apache License, Version 2.0
'''

import numpy as np
import cv2
import os
import sys
import argparse
import glob
import subprocess
import signal
from timeit import default_timer as timer
import time
from datetime import datetime
import getpass
import socket
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GObject, GstBase, GstVideo, GstApp, GLib

# NNStreamer Python API (if used)
try:
    import nnstreamer_python as nns
except ImportError:
    print("NNStreamer Python API not found. Running in GStreamer-only mode.")

# Local imports
sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

# Constants
SCALE = 1.0
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.75 * SCALE
TEXT_COLOR = (0, 0, 255)
TEXT_THICKNESS = max(1, int(2 * SCALE))
LINE_TYPE = cv2.LINE_AA

class GstDisplay:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        
        # Create GStreamer display pipeline
        self.pipeline = Gst.Pipeline.new("display-pipeline")
        
        # Elements
        self.appsrc = Gst.ElementFactory.make("appsrc", "source")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "convert")
        self.sink = Gst.ElementFactory.make("autovideosink", "sink")
        
        if not all([self.pipeline, self.appsrc, self.videoconvert, self.sink]):
            raise RuntimeError("Failed to create GStreamer elements")
        
        # Configure appsrc
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height},framerate=30/1")
        self.appsrc.set_property("caps", caps)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", True)
        
        # Add and link elements
        self.pipeline.add(self.appsrc)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.sink)
        
        self.appsrc.link(self.videoconvert)
        self.videoconvert.link(self.sink)
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
    
    def push_frame(self, frame):
        """Push frame to GStreamer display pipeline"""
        if frame.shape != (self.height, self.width, 3):
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Create GStreamer buffer from numpy array (zero-copy if possible)
        buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        self.appsrc.emit("push-buffer", buffer)
    
    def close(self):
        """Cleanup"""
        self.pipeline.set_state(Gst.State.NULL)

class OptimizedBlazePose:
    def __init__(self, args):
        self.args = args
        self.width = 640
        self.height = 480
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        
        self.fps_count = 0

        # Initialize models
        self.initialize_models()
        
        # Setup GStreamer pipeline
        self.create_pipeline()
        
        # Pre-allocated buffers (reduces memory allocation in callbacks)
        self.frame_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.display = GstDisplay(self.width, self.height)
        
        # Profiling setup
        self.profile_csv = './blaze_detect_nnstreamer_optimized.csv'
        if self.args.profilelog:
            if os.path.isfile(self.profile_csv):
                self.f_profile = open(self.profile_csv, "a")
            else:
                self.f_profile = open(self.profile_csv, "w")
                self.f_profile.write("timestamp,user,host,pipeline,resize,detector_pre,detector_model,"
                                   "detector_post,extract_roi,landmark_pre,landmark_model,"
                                   "landmark_post,annotate,total,fps\n")
    
    def initialize_models(self):
        """Initialize BlazePose models based on args"""
        if self.args.blaze == "hand":
            self.detector_type = "blazepalm"
            self.landmark_type = "blazehandlandmark"
            default_detector = 'models/palm_detection_lite.tflite'
            default_landmark = 'models/hand_landmark_lite.tflite'
        elif self.args.blaze == "face":
            self.detector_type = "blazeface"
            self.landmark_type = "blazefacelandmark"
            default_detector = 'models/face_detection_short_range_pixabay1675_unsigned_uint8quant_vela.tflite'
            default_landmark = 'models/face_landmark_pixabay1941_unsigned_uint8quant_vela.tflite'
        elif self.args.blaze == "pose":
            self.detector_type = "blazepose"
            self.landmark_type = "blazeposelandmark"
            default_detector = "models/pose_detection.tflite"
            default_landmark = "models/pose_landmark_full.tflite"
        
        # Set model paths
        self.detector_model = self.args.model1 if self.args.model1 else default_detector
        self.landmark_model = self.args.model2 if self.args.model2 else default_landmark
        
        # Setup delegate
        delegate = None
        if self.args.npu and "_vela" not in self.detector_model and "_vela" not in self.landmark_model:
            delegate = "/usr/lib/libethosu_delegate.so"
        
        # Initialize detectors
        self.detector = BlazeDetector(self.detector_type, delegate_path=delegate)
        self.detector.set_debug(debug=self.args.debug)
        self.detector.load_model(self.detector_model)
        
        self.landmark = BlazeLandmark(self.landmark_type, delegate_path=delegate)
        self.landmark.set_debug(debug=self.args.debug)
        self.landmark.load_model(self.landmark_model)
    
    def create_pipeline(self):
        """Create optimized GStreamer pipeline"""
        # Find video device
        dev_video = get_video_dev_by_name("uvcvideo")
        input_source = self.args.input if self.args.input else dev_video if dev_video else "/dev/video0"
        
        pipeline_str = (
            f"v4l2src device={input_source} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "imxvideoconvert_pxp ! video/x-raw,format=BGR ! "
            "tee name=t ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "appsink name=appsink emit-signals=true sync=false max-buffers=2 drop=true "
            "t. ! queue ! videoconvert ! videoflip method=counterclockwise ! autovideosink sync=false"
        )
        
        print(f"GStreamer Pipeline:\n{pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Configure appsink
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.connect("new-sample", self.on_new_sample)
        
        # Bus for message handling
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_message)
    
    def on_new_sample(self, appsink):
        """Optimized callback for new frames"""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        # Use pre-allocated buffer (faster than new allocation)
        self.frame_buffer[:,:,:] = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
            (self.height, self.width, 3))
        
        # Process frame
        self.process_frame(self.frame_buffer)
        
        buffer.unmap(map_info)
        
        # Update FPS counter
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            now = time.time()
            self.fps = 10.0 / (now - self.last_time)
            self.last_time = now
        # if self.frame_count == 10:
        #     rt = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        #     rt_fps_valid = True
        #     rt_fps = 10.0/t
        #     rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #     rt_fps_count = 0
            
        return Gst.FlowReturn.OK
    
    def process_frame(self, frame):
        """Process frame through BlazePose models"""
        # Initialize profiling variables if needed
        if self.args.profilelog or self.args.profileview:
            start_total = timer()
            profile_resize = 0
            profile_extract = 0
            profile_annotate = 0
        
        image = frame.copy()
        output = frame.copy()

        # Detection pipeline
        if self.args.profilelog or self.args.profileview:
            start = timer()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img1, scale1, pad1 = self.detector.resize_pad(image)
        
        if self.args.profilelog or self.args.profileview:
            profile_resize = timer() - start
        
        normalized_detections = self.detector.predict_on_image(img1)
        
        if len(normalized_detections) > 0:
            # Denormalize detections
            if self.args.profilelog or self.args.profileview:
                start = timer()
            detections = self.detector.denormalize_detections(normalized_detections, scale1, pad1)
            xc, yc, scale, theta = self.detector.detection2roi(detections)
            roi_img, roi_affine, roi_box = self.landmark.extract_roi(image, xc, yc, theta, scale)
            
            if self.args.profilelog or self.args.profileview:
                profile_extract = timer() - start
            
            # Landmark prediction
            flags, normalized_landmarks = self.landmark.predict(roi_img)
            
            # Annotate
            if self.args.profilelog or self.args.profileview:
                start = timer()
            
            landmarks = self.landmark.denormalize_landmarks(normalized_landmarks, roi_affine)
            
            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                #if True: #flag>.5:
                if self.landmark_type == "blazehandlandmark":
                    draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)
                elif self.landmark_type == "blazefacelandmark":
                    draw_landmarks(output, landmark[:,:2], FACE_CONNECTIONS, size=1)                                    
                elif self.landmark_type == "blazeposelandmark":
                    if landmarks.shape[1] > 33:
                        draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                    else:
                        draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)    
            
            draw_roi(output, roi_box)
            draw_detections(output, detections)
            
            if self.args.profilelog or self.args.profileview:
                profile_annotate = timer() - start
        
        # Display FPS if enabled
        if self.args.fps and self.fps > 0:
            fps_text = f"FPS: {self.fps:.2f}"
            cv2.putText(output, fps_text, (10, self.height-10), 
                       TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS, LINE_TYPE)
        
        # Push to display
        self.display.push_frame(output)
        
        # Log profiling data if enabled
        if self.args.profilelog:
            self.log_profile_data(profile_resize, profile_extract, profile_annotate, start_total)
    
    
    
    def log_profile_data(self, resize_time, extract_time, annotate_time, start_time):
        """Log profiling data to CSV"""
        total_time = timer() - start_time
        timestamp = datetime.now().isoformat()
        user = getpass.getuser()
        host = socket.gethostname()
        
        csv_line = (
            f"{timestamp},{user},{host},nnstreamer,"
            f"{resize_time},{self.detector.profile_pre},{self.detector.profile_model},"
            f"{self.detector.profile_post},{extract_time},{self.landmark.profile_pre},"
            f"{self.landmark.profile_model},{self.landmark.profile_post},"
            f"{annotate_time},{total_time},{self.fps if self.fps > 0 else 0}\n"
        )
        
        self.f_profile.write(csv_line)
        self.f_profile.flush()
           
    
    def on_message(self, bus, message):
        """Handle GStreamer bus messages"""
        if message.type == Gst.MessageType.EOS:
            print("End of stream")
            self.stop()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err} - {debug}")
            self.stop()
    
    def start(self):
        """Start the pipeline"""
        self.running = True
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Main loop (using GLib for proper integration)
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Cleanup"""
        if self.running:
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
            if self.display:
                self.display.close()
            if hasattr(self, 'f_profile'):
                self.f_profile.close()
            if hasattr(self, 'loop'):
                self.loop.quit()
            sys.exit(0)
            
def get_video_dev_by_name(src):
    """Find video device by name"""
    devices = glob.glob("/dev/video*")
    for dev in sorted(devices):
        proc = subprocess.run(['v4l2-ctl', '-d', dev, '-D'], 
                            capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev
def parse_args():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type=str, default="", help="Video input device. Default is auto-detect (first usbcam)")
    ap.add_argument('-I', '--image', default=False, action='store_true', help="Use 'woman_hands.jpg' image as input. Default is usbcam")
    ap.add_argument('-b', '--blaze', type=str, default="hand", help="Application (hand, face, pose). Default is hand")
    ap.add_argument('-m', '--model1', type=str, help='Path of blazepalm model. Default is models/palm_detection_without_custom_op.tflite')
    ap.add_argument('-n', '--model2', type=str, help='Path of blazehandlardmark model. Default is models/hand_landmark.tflite')
    ap.add_argument('-d', '--debug', default=False, action='store_true', help="Enable Debug mode. Default is off")
    ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
    ap.add_argument('-z', '--profilelog', default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
    ap.add_argument('-Z', '--profileview', default=False, action='store_true', help="Enable Profile View (Latency). Default is off")
    ap.add_argument('-f', '--fps', default=False, action='store_true', help="Enable FPS display. Default is off")
    ap.add_argument('-N', '--npu', default=False, action='store_true', help="Enable NPU")
    return ap.parse_args()

def main():
    """Entry point"""
    # Initialize GStreamer
    Gst.init(None)
    
    # Parse args
    args = parse_args()
    
    # Create and run pipeline
    pipeline = OptimizedBlazePose(args)
    
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, lambda s, f: pipeline.stop())
    
    # Start processing
    pipeline.start()

if __name__ == "__main__":
    main()