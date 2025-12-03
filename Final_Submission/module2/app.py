import os
import cv2
import glob
import numpy as np
import shutil
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Calculate the absolute path to THIS folder (module2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths for all folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'dataset')
SCENE_SOURCE = os.path.join(TEMPLATES_DIR, 'test_scene.jpg')

MATCH_THRESHOLD = 0.35

# Files to ignore
IGNORE_FILES = [
    'test_scene.jpg', 
    'task2_source.jpg', 
    'task1_result.jpg', 
    'result.jpg',
    '.DS_Store'
]

# Ensure the static folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_display = None
    processed_display = None

    # 1. Setup Original Image
    if os.path.exists(SCENE_SOURCE):
        dest_path = os.path.join(UPLOAD_FOLDER, 'original_scene.jpg')
        shutil.copy(SCENE_SOURCE, dest_path)
        original_display = url_for('static', filename='original_scene.jpg')
    else:
        print(f"ERROR: Could not find source image at {SCENE_SOURCE}")

    # 2. Handle run button
    if request.method == 'POST':
        if os.path.exists(SCENE_SOURCE):
            process_image_and_blur(SCENE_SOURCE)
            processed_display = url_for('static', filename='result.jpg')

    return render_template('index.html', original=original_display, processed=processed_display)

def process_image_and_blur(image_path):
    main_img = cv2.imread(image_path)
    if main_img is None:
        print("Error reading main image")
        return

    gray_main = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    final_img = main_img.copy()

    # Use the absolute TEMPLATES_DIR
    all_files = glob.glob(os.path.join(TEMPLATES_DIR, "*"))
    
    print(f"Processing Scene: {image_path}")
    
    for t_path in all_files:
        t_name = os.path.basename(t_path)
        
        if t_name in IGNORE_FILES or not t_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        template = cv2.imread(t_path)
        if template is None: continue
        
        gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        (tH, tW) = gray_temp.shape[:2]

        if tH > gray_main.shape[0] or tW > gray_main.shape[1]:
            continue

        best_match = None
        # Multi-Scale Detection
        for scale in np.linspace(0.8, 1.2, 10):
            resized_w = int(tW * scale)
            resized_h = int(tH * scale)
            
            if resized_w > gray_main.shape[1] or resized_h > gray_main.shape[0]:
                continue

            resized_t = cv2.resize(gray_temp, (resized_w, resized_h))
            res = cv2.matchTemplate(gray_main, resized_t, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(res)

            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, resized_w, resized_h)

        if best_match:
            (score, (x, y), w, h) = best_match
            
            if score >= MATCH_THRESHOLD:
                print(f"Blurring {t_name} ({int(score*100)}%)")
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                
                # Blur Logic
                roi = final_img[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                final_img[y:y+h, x:x+w] = blurred_roi
                
                cv2.rectangle(final_img, top_left, bottom_right, (0, 0, 255), 3)

    # Save result to the absolute static path
    result_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(result_path, final_img)
    print(f"Saved result to {result_path}")

if __name__ == '__main__':
    app.run(debug=True)