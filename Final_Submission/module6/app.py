from flask import Flask, render_template, Response, request, jsonify
import cv2
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tracking_strategies import ArucoStrategy, CSRTStrategy, SAM2Strategy

app = Flask(__name__)

class CameraContext:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.strategy = ArucoStrategy() 
        self.frame_counter = 0

    def switch_strategy(self, mode):
        self.frame_counter = 0 
        if mode == 'marker':
            self.strategy = ArucoStrategy()
        elif mode == 'markerless':
            self.strategy = CSRTStrategy()
        elif mode == 'sam2':
            self.strategy = SAM2Strategy()
        print(f"Switched to strategy: {mode}")

    def get_feed(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                processed_frame = self.strategy.update(frame, self.frame_counter)
                self.frame_counter += 1
                
                if processed_frame is None:
                    continue

                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error in strategy: {e}")
                continue

cam_context = CameraContext()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(cam_context.get_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_mode', methods=['POST'])
def select_mode():
    mode = request.form.get('mode')
    cam_context.switch_strategy(mode)
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)