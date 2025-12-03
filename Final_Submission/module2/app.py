import os
import cv2
import glob
import numpy as np
import shutil
import traceback
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# setting up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'dataset')
SCENE_SOURCE = os.path.join(TEMPLATES_DIR, 'test_scene.jpg')

# --- TUNED SETTINGS ---
# 0.25 allows faint objects (remote) to be found
MATCH_THRESHOLD = 0.25  
# 1200px gives enough detail for the remote buttons without crashing
MAX_WIDTH = 1200        

IGNORE_FILES = ['test_scene.jpg', 'task2_source.jpg', 'task1_result.jpg', 'result.jpg', '.DS_Store']

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def resize_image_and_get_ratio(image, target_width):
    if image is None: return None, 1.0
    h, w = image.shape[:2]
    if w <= target_width: return image, 1.0
    
    scale = target_width / w
    resized = cv2.resize(image, (target_width, int(h * scale)))
    return resized, scale

@app.route('/', methods=['GET', 'POST'])
def index():
    original_display = None
    processed_display = None

    try:
        if os.path.exists(SCENE_SOURCE):
            dest_path = os.path.join(UPLOAD_FOLDER, 'original_scene.jpg')
            shutil.copy(SCENE_SOURCE, dest_path)
            original_display = url_for('static', filename='original_scene.jpg')

        if request.method == 'POST':
            if os.path.exists(SCENE_SOURCE):
                print("Starting processing...")
                process_image_and_blur(SCENE_SOURCE)
                processed_display = url_for('static', filename='result.jpg')
                print("Processing done.")

    except Exception as e:
        print("CRASH PREVENTED IN MODULE 2:")
        traceback.print_exc()

    return render_template('index.html', original=original_display, processed=processed_display)

def process_image_and_blur(image_path):
    try:
        main_img = cv2.imread(image_path)
        if main_img is None: return

        # resize scene to save RAM
        main_img, ratio = resize_image_and_get_ratio(main_img, MAX_WIDTH)
        
        gray_main = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
        final_img = main_img.copy()

        all_files = glob.glob(os.path.join(TEMPLATES_DIR, "*"))
        
        for t_path in all_files:
            t_name = os.path.basename(t_path)
            if t_name in IGNORE_FILES or not t_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            template = cv2.imread(t_path)
            if template is None: continue
            
            # shrink template to match the new scene size
            if ratio < 1.0:
                h, w = template.shape[:2]
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                if new_w > 0 and new_h > 0:
                    template = cv2.resize(template, (new_w, new_h))

            gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            (tH, tW) = gray_temp.shape[:2]

            if tH > gray_main.shape[0] or tW > gray_main.shape[1]:
                continue

            best_match = None
            
            # Restricted range (0.8 to 1.2) stops "giant/tiny" false positives
            # 20 steps ensures we hit the exact size needed for the remote
            for scale in np.linspace(0.8, 1.2, 20): 
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
                    print(f"Match: {t_name} ({score:.2f})")
                    roi = final_img[y:y+h, x:x+w]
                    
                    blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
                    final_img[y:y+h, x:x+w] = blurred_roi
                    
                    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        result_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
        cv2.imwrite(result_path, final_img)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    app.run(debug=True)