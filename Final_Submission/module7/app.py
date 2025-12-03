from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import mediapipe as mp
import csv
import time
import math
import os
import traceback

app = Flask(__name__)

# setting up paths so we know where to save the csv file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, 'pose_data.csv')

# camera calibration values from my experiment
ORIG_FX = 1102.12  
ORIG_FY = 1105.95
ORIG_CX = 618.990
ORIG_CY = 342.080
CALIB_W = 1280
CALIB_H = 720
BASELINE = 10.0 # distance between the two camera positions

# setting up mediapipe tools for detection and drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

# helper to adjust camera numbers if the image resolution changes
def get_scaled_intrinsics(img_w, img_h):
    scale_x = img_w / CALIB_W
    scale_y = img_h / CALIB_H
    fx = ORIG_FX * scale_x
    fy = ORIG_FY * scale_y
    cx = ORIG_CX * scale_x
    cy = ORIG_CY * scale_y
    return fx, fy, cx, cy

@app.route('/calculate_stereo', methods=['POST'])
def calculate_stereo():
    try:
        data = request.get_json()
        p_left, p_right = data['p_left'], data['p_right']
        
        # get correct focal length for the current image size
        fx, _, _, _ = get_scaled_intrinsics(data['img_w'], data['img_h'])
        
        # disparity is just how far the point moved between left and right images
        disparity = abs(p_left['x'] - p_right['x'])
        
        if disparity < 1.0:
            return jsonify({"error": "You clicked the same pixel"})

        # standard stereo vision formula: Z = (f * B) / d
        calculated_Z = (fx * BASELINE) / disparity
        print(f"Stereo: Disparity={disparity:.2f} px | Z={calculated_Z:.2f} cm")
        return jsonify({"Z": round(calculated_Z, 4), "disparity": round(disparity, 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/calculate_dist', methods=['POST'])
def calculate_dist():
    try:
        data = request.get_json()
        p1, p2, Z = data['p1'], data['p2'], data['Z']
        fx, fy, cx, cy = get_scaled_intrinsics(data['img_w'], data['img_h'])

        # convert 2d pixel points to 3d real-world coordinates using depth Z
        X1 = (p1['x'] - cx) * Z / fx
        Y1 = (p1['y'] - cy) * Z / fy
        X2 = (p2['x'] - cx) * Z / fx
        Y2 = (p2['y'] - cy) * Z / fy

        # simple distance formula in 3d space
        dist = math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
        return jsonify({"dist_cm": round(dist, 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def generate_frames():
    # opening the webcam
    cap = cv2.VideoCapture(0)
    
    # create a fresh csv file with headers
    try:
        with open(CSV_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'nose_x', 'nose_y', 'right_wrist_x', 'right_wrist_y'])
    except Exception as e:
        print(f"CSV Init Error: {e}")

    # using mediapipe holistic model to track body and hands
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = cap.read()
            if not success: break

            # mediapipe needs rgb images, but opencv gives bgr
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # draw the stick figure lines on top of the video
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # try to save the keypoints to the csv
            try:
                row = [time.time()]
                
                # save nose coordinates if found
                if results.pose_landmarks:
                    nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                    row.extend([nose.x, nose.y])
                else: row.extend([0, 0])

                # save right wrist coordinates if found
                if results.right_hand_landmarks:
                    wrist = results.right_hand_landmarks.landmark[0]
                    row.extend([wrist.x, wrist.y])
                else: row.extend([0, 0])
                
                # write the row to the file
                with open(CSV_FILE_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except: pass

            # encode the frame to jpg so the browser can display it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    # standard flask way to stream the video frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv')
def download_csv():
    # lets the user download the data we recorded
    if os.path.exists(CSV_FILE_PATH):
        return send_file(CSV_FILE_PATH, as_attachment=True)
    else:
        return "File not found. Please run Pose Tracking first!", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)