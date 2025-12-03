# Assignment 2: Object Detection & Deblurring

## Description
This assignment covers two main computer vision tasks:
1. **Object Detection:** Using Template Matching to find objects like a watch, wallet, or keys in a cluttered scene and automatically blurring them for privacy. This part of the assignment first takes images of the objects, and uses these images to deetct the same objects from a cluttered environment. The full webpage implementation implements this aspect as well as blurring to create a privacy blur for security.
2. **Image Restoration:** Using Fourier Transform to recover a sharp image from a blurred one.

## Files
* `app.py`: Web application for the Privacy Redaction System (Task 1 & 3).
* `task1.py`: Standalone script for testing template matching logic.
* `task2.py`: Script for Task 2 that performs Fourier Transform deblurring and displays the result.
* `dataset/`: Contains the scene image and templates for 10+ objects.

## How to Run Object Detection
First ensure you are in the correct module2.
1. ```bash
      python task1.py
2. Open the browser to http://127.0.0.1:5000
## How to Run Image Restoration
1. ```bash
      python task2.py
2. Open the browser to http://127.0.0.1:5000
## To open webpage
1. ```bash
      python app.py
2. Open the browser to http://127.0.0.1:5000
3. Follow instructions on the webpage.
