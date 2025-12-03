# Assignment 5-6: Real-Time Object Tracking

## Description
This module implements three different object tracking strategies:
1. **Mode A (Marker):** Tracks standard ArUco markers (4x4).
2. **Mode B (Markerless):** Uses the CSRT algorithm to track any user-selected object.
3. **Mode C (SAM2):** Demonstrates the SAM2 using a pre-processed video and segmentation masks.

## Files
* `app.py`: Main app that streams the video feed and handles mode switching.
* `tracking_strategies.py`: Contains the logic for `ArucoStrategy`, `CSRTStrategy`, and `SAM2Strategy`.
* `static/sam2_demo_video.mp4`: The video file for the SAM2 demo.
* `static/segmentation.npz`: The segmentation data for the SAM2 demo.

## How to Run
First ensure you are in module6
1. Ensure the `static` folder contains the `.mp4` and `.npz` files.
2. Run the app:
   ```bash
   python app.py
3. Open the browser to http://127.0.0.1:5000
4. Follow instructions on webpage.
This live webcam does not function on cloud hosting servers as they do not have direct access to the webcam, but runs perfectly on localhost 5050.
