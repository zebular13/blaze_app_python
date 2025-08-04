import os
import subprocess
import threading
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Working pipeline with proper syntax
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

def start_pipeline():
    global process
    env = os.environ.copy()
    env.update({'GST_DEBUG': '2'})
    
    process = subprocess.Popen(
        ["gst-launch-1.0", "-v"] + PIPELINE.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0
    )
    threading.Thread(target=monitor_stderr, daemon=True).start()

def monitor_stderr():
    while True:
        line = process.stderr.readline()
        if line:
            print("GSTERR:", line.decode().strip())
        elif process.poll() is not None:
            break

def generate():
    while True:
        if process.poll() is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   b'Pipeline not running' + b'\r\n')
            continue
            
        try:
            # Read JPEG frame markers
            header = process.stdout.read(2)
            if header != b'\xff\xd8':
                continue
                
            # Read until JPEG end marker
            data = header
            while b'\xff\xd9' not in data:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                data += chunk
            
            if b'\xff\xd9' in data:
                frame = data[:data.index(b'\xff\xd9')+2]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame + b'\r\n')
        except Exception as e:
            print("Frame error:", str(e))

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head><title>Camera Feed</title></head>
        <body>
            <h1>Camera Feed</h1>
            <img src="{{ url_for('video_feed') }}">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    start_pipeline()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        if process:
            process.terminate()