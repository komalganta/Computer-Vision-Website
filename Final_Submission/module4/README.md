# Assignment 4: Image Stitching & SIFT

## Description
This assignment implements:
1. **Panorama Stitching:** Combines 4 overlapping images into a single panoramic view. It also includes a manual stitching fallback if the automatic mode fails.
2. **SIFT Feature Extraction:** My custom implementation from scatch of SIFT logic using DoG and keypoints compared against OpenCV's built-in SIFT.

## Files
* `app.py`: Flask app handling the stitching and SIFT logic.
* `static/images/`: Contains the source images (`1.jpg` to `4.jpg`) and a phone panorama for comparison (`phone.jpg`).
* `templates/assignment4.html`: The interface to trigger the algorithms.

## How to Run
Ensure you are in module4.
1. Run the script:
   ```bash
   python app.py
2. Open the browser to http://127.0.0.1:5000
3. Follow instructions on webpage.