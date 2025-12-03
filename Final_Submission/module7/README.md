# Assignment 7: Calibrated Stereo & Pose Tracking

## Description
This assignment has two main parts:
1. **Calibrated Stereo Vision:** Calculates the real-world depth (`Z = 60cm`) and size of an object using two images (Left/Right eyes) and a known baseline distance (`10 cm`). The measurements of the object are `width=24.3cm` and `length=29.1cm`.
2. **Pose Estimation:** Tracks the full body and hand in real-time using MediaPipe and saves the coordinates to a CSV file.

## Files
* `app.py`: Main Flask application that handles the stereo math and video streaming.
* `pose_tracking.py`: Standalone script if you want to run tracking without the web interface.
* `static/left_img.jpg` & `right_img.jpg`: The stereo image pair used for measurement.
* `pose_data.csv`: The output file where the tracked landmarks are saved.

## How to Run
Ensure you are located in module7 first.
1. 
   ```bash
   python app.py
2. Open the browser to http://127.0.0.1:5000
3. Follow instructions in the webpage.