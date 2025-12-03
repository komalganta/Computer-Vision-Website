# Assignment 3: Features, Edges & Segmentation

## Description
This module analyzes object features and boundaries. It performs:
* **Part 1:** Gradient magnitude and angle visualization + Laplacian of Gaussian (LoG).
* **Part 2:** Keypoint detection using Canny Edges and Harris Corners.
* **Part 3:** Object boundary detection using contours.
* **Part 4:** Segmentation of a non-rectangular object (a shoe) using ArUco markers.

## Files
* `app.py`: The main Flask app that processes the images and serves the results.
* `dataset/`: Contains 10 images of a firestick for feature detection.
* `dataset_aruco/`: Contains 10 images of a shoe with ArUco markers for segmentation.
* `templates/`: HTML files for viewing the results.

## How to Run
Ensure that you are in module3.
1. Run the application:
   ```bash
   python app.py
2. Open the browser to http://127.0.0.1:5000
3. Instructions provided in the webpage once opened.