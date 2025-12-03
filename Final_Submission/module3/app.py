import cv2
import numpy as np
import os
import glob
from flask import Flask, render_template, url_for, abort

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
ARUCO_DATASET_DIR = os.path.join(BASE_DIR, 'dataset_aruco')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_save_image(filename):
    input_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(input_path):
        return None
    
    output_name_base = os.path.splitext(filename)[0]
    
    output_paths = {
        "original": os.path.join(OUTPUT_DIR, f"{output_name_base}_original.jpg"),
        "magnitude": os.path.join(OUTPUT_DIR, f"{output_name_base}_magnitude.jpg"),
        "angle": os.path.join(OUTPUT_DIR, f"{output_name_base}_angle.jpg"),
        "log": os.path.join(OUTPUT_DIR, f"{output_name_base}_log.jpg"),
        "edges": os.path.join(OUTPUT_DIR, f"{output_name_base}_edges.jpg"),
        "corners": os.path.join(OUTPUT_DIR, f"{output_name_base}_corners.jpg"),
        "boundary": os.path.join(OUTPUT_DIR, f"{output_name_base}_boundary.jpg"),
    }

    # check if done
    if os.path.exists(output_paths["boundary"]):
        return output_name_base
        
    frame = cv2.imread(input_path)
    if frame is None:
        return None

    # resize and grayscale
    h, w = 360, 480
    frame_small = cv2.resize(frame, (w, h))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(output_paths["original"], frame_small)

    # part 1: gradients
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely)
    log = cv2.Laplacian(blur, cv2.CV_64F, ksize=5)

    # save part 1 images
    mag_display = cv2.convertScaleAbs(magnitude)
    angle_display = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    log_display = cv2.convertScaleAbs(log)
    
    cv2.imwrite(output_paths["magnitude"], mag_display)
    cv2.imwrite(output_paths["angle"], angle_display)
    cv2.imwrite(output_paths["log"], log_display)
    
    # part 2: keypoints
    # lower thresholds to catch dark edges
    canny_edges = cv2.Canny(blur, 30, 100)
    cv2.imwrite(output_paths["edges"], canny_edges)

    # harris corners
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corner_image = frame_small.copy()
    corner_image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite(output_paths["corners"], corner_image)

    # part 3: boundary
    # close gaps in the edges so we get a closed loop
    kernel = np.ones((5,5), np.uint8)
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
    
    # find contours from the closed edges
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundary_image = frame_small.copy()
    
    if contours:
        # get the biggest one
        largest_contour = max(contours, key=cv2.contourArea)
        
        # ignore small noise
        if cv2.contourArea(largest_contour) > 500:
            cv2.drawContours(boundary_image, [largest_contour], -1, (0, 255, 0), 3)

    cv2.imwrite(output_paths["boundary"], boundary_image)

    return output_name_base

# part4

def process_and_save_aruco(filename):
    input_path = os.path.join(ARUCO_DATASET_DIR, filename)
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find ArUco image at {input_path}")
        return None
    
    output_name_base = os.path.splitext(filename)[0] + "_aruco"
    
    output_paths = {
        "original": os.path.join(OUTPUT_DIR, f"{output_name_base}_original.jpg"),
        "segmented": os.path.join(OUTPUT_DIR, f"{output_name_base}_segmented.jpg"),
    }
    if os.path.exists(output_paths["segmented"]):
        return output_name_base
        
    frame = cv2.imread(input_path)
    if frame is None: return None
    
    frame = cv2.resize(frame, (640, 480))
    cv2.imwrite(output_paths["original"], frame)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(frame)
    
    segmented_image = frame.copy()
    
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(segmented_image, corners, ids)
        
        all_marker_corners = np.concatenate(corners)
        all_marker_corners = all_marker_corners.reshape(-1, 2)
        all_marker_corners = all_marker_corners.astype(np.int32)
        
        hull = cv2.convexHull(all_marker_corners)
        cv2.drawContours(segmented_image, [hull], -1, (0, 255, 0), 3)
    
    cv2.imwrite(output_paths["segmented"], segmented_image)
    
    return output_name_base

@app.route('/')
def index():
    #Home page for Parts 1-3

    image_paths = glob.glob(os.path.join(DATASET_DIR, '*.[jp][pn]g'))
    image_filenames = [os.path.basename(p) for p in image_paths]
    return render_template('index.html', filenames=image_filenames)

@app.route('/view/<filename>')
def view_image(filename):
    #Results page for Parts 1-3
    output_name = process_and_save_image(filename)
    if output_name is None:
        return abort(404, "Image not found in dataset.")
    return render_template('result.html', output_name=output_name)

@app.route('/aruco')
def index_aruco():
    #Home page for Part 4
    image_paths = glob.glob(os.path.join(ARUCO_DATASET_DIR, '*.[jp][pn]g'))
    image_filenames = [os.path.basename(p) for p in image_paths]
    return render_template('aruco_index.html', filenames=image_filenames)

@app.route('/view_aruco/<filename>')
def view_aruco(filename):
    #Results page for Part 4
    output_name = process_and_save_aruco(filename)
    if output_name is None:
        return abort(404, "ArUco image not found in dataset.")
    return render_template('aruco_result.html', output_name=output_name)


if __name__ == '__main__':
    app.run(debug=True)