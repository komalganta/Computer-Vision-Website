import cv2
import numpy as np
import os
import glob

SCENE_PATH = 'dataset/test_scene.jpg'
TEMPLATES_DIR = 'dataset'             
OUTPUT_FILENAME = 'static/task1_result.jpg'
MATCH_THRESHOLD = 0.3

# files here that are not object templates
IGNORE_FILES = [
    'test_scene.jpg',       
    'task2_source.jpg',    
    'task1_result.jpg', 
    'result.jpg',
    '.DS_Store'           
]

def run_detection():
    print("--- Task 1 Started ---")

    if not os.path.exists(SCENE_PATH):
        print(f"Error: {SCENE_PATH} not found.")
        return
    
    main_img = cv2.imread(SCENE_PATH)
    if main_img is None:
        print("Error: Could not read test_scene.jpg")
        return

    gray_main = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    result_img = main_img.copy()

    # 1. Get all valid templates from dataset folder
    all_files = glob.glob(os.path.join(TEMPLATES_DIR, "*"))
    valid_templates = []

    for f_path in all_files:
        f_name = os.path.basename(f_path)
        # Check if it is an image and not in ignore list
        if f_name.lower().endswith(('.jpg', '.png', '.jpeg')) and f_name not in IGNORE_FILES:
            valid_templates.append(f_path)

    print(f"Scanning with {len(valid_templates)} templates: {[os.path.basename(t) for t in valid_templates]}")

    # 2. Loop through valid templates
    for t_path in valid_templates:
        t_name = os.path.basename(t_path).split('.')[0]
        
        template = cv2.imread(t_path)
        if template is None: continue

        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        (tH, tW) = gray_template.shape[:2]

        if tH > gray_main.shape[0] or tW > gray_main.shape[1]:
            print(f"Skipping {t_name}: Template is larger than scene.")
            continue

        best_match = None

        for scale in np.linspace(0.8, 1.2, 10):
            resized_w = int(tW * scale)
            resized_h = int(tH * scale)
            
            if resized_w > gray_main.shape[1] or resized_h > gray_main.shape[0]:
                continue

            resized_t = cv2.resize(gray_template, (resized_w, resized_h))
            res = cv2.matchTemplate(gray_main, resized_t, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(res)

            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, resized_w, resized_h)

        if best_match:
            (score, (x, y), w, h) = best_match
            if score >= MATCH_THRESHOLD:
                print(f"[MATCH] {t_name}: {int(score*100)}%")
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(result_img, t_name, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save and Show
    cv2.imwrite(OUTPUT_FILENAME, result_img)
    print(f"Saved result to {OUTPUT_FILENAME}")

    # Resize for display so fits on screen
    h, w = result_img.shape[:2]
    small = cv2.resize(result_img, (int(w*0.3), int(h*0.3)))
    cv2.imshow("Task 1 Result", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()