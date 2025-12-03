# Assignment 1: Real-World Dimension Measurement

## Description
This project implements perspective projection to measure the real-world size of objects from a 2D image. It uses a reference object (Rubik's cube) at a known distance (`Z = 34.0 cm`) to calibrate the camera intrinsics (`fx`, `fy`). Users can click on two points in the image to measure the distance between them in centimeters. The real width and height measurements of the rubix cube in the module is 5.5cm.

## Files
* `app.py`: Main Flask application that handles the web interface and calculation logic.
* `measure_world.py`: Standalone script for testing the projection math.
* `templates/index.html`: The frontend interface for clicking points.
* `static/rubixcube1.jpg`: The test image used for validation.

## How to Run
1. Make sure you are in the `module1` folder.
2. Run the app:
   ```bash
   python app.py
3. Open the browser to http://127.0.0.1:5000
3. Once you run the module, instructions are provided on the page.