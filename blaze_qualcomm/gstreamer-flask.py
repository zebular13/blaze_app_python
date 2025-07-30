import subprocess
import threading
from flask import Flask, Response

app = Flask(__name__)

# GStreamer pipeline that outputs JPEG frames to stdout
PIPELINE = (
    "v4l2src device=/dev/video2 ! "
    "image/jpeg,width=640,height=480,framerate=30/1 ! "
    "jpegdec ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "jpegenc quality=85 ! "
    "fdsink fd=1"
)

process = None
running = False

def start_gstreamer():
    global process, running
    running = True
    try:
        process = subprocess.Popen(
            ["gst-launch-1.0", "-q"] + PIPELINE.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        print("GStreamer pipeline started")
        while running:
            if process.poll() is not None:
                print("GStreamer process ended unexpectedly")
                break
    except Exception as e:
        print(f"Error starting pipeline: {e}")
    finally:
        if process:
            process.terminate()
            process.wait()

def generate():
    while True:
        if not process or process.poll() is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   b'Camera not available' + b'\r\n')
            continue
            
        # Read JPEG frame from stdout
        # GST header is 16 bytes: [0xff, 0xd8, ..., 0xff, 0xd9]
        # We'll read until we find the JPEG end marker
        data = b''
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            data += chunk
            if b'\xff\xd9' in data:  # JPEG end marker
                frame = data[:data.index(b'\xff\xd9')+2]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                break

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Webcam Feed</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            img { border: 1px solid #ccc; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Webcam Feed</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    global running
    running = False
    if process:
        process.terminate()
        process.wait()

if __name__ == '__main__':
    try:
        # Start GStreamer in a separate thread
        threading.Thread(target=start_gstreamer, daemon=True).start()
        
        # Start Flask app
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cleanup()