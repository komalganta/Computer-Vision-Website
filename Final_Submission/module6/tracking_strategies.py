import cv2
import numpy as np
from abc import ABC, abstractmethod
import os

# making a template for my trackers so they all follow the same rules
class TrackerStrategy(ABC):
    @abstractmethod
    def update(self, frame, frame_count):
        # every tracker needs to take a frame and return it processed
        pass

# strategy 1: ArUco Marker Tracker
# this one looks for those square QR-like markers
class ArucoStrategy(TrackerStrategy):
    def __init__(self):
        # setting up the dictionary so it knows which markers to look for (4x4)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()

    def update(self, frame, frame_count):
        if frame is None: return frame
        
        # magic function that finds the corners and IDs of markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.params)
        
        if ids is not None:
            # if we found any markers, loop through them
            for corner_set in corners:
                # convert corners to int so we can draw them
                pts = corner_set[0].astype(np.int32)
                
                # draw a cool neon green box around the marker
                cv2.polylines(frame, [pts], True, (57, 255, 20), 3)
                
                # find the center point just for fun
                c_x = int(np.mean(pts[:, 0]))
                c_y = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (c_x, c_y), 5, (0, 0, 255), -1)
                
        return frame

# strategy 2: CSRT Tracker
# this lets the user track any random object
class CSRTStrategy(TrackerStrategy):
    def __init__(self):
        self.tracker = None
        self.initialized = False
        self.bbox_color = (255, 0, 255)

    def init_tracker(self, frame):
        # grabbing the frame size to find the center
        h, w = frame.shape[:2]
        roi_size = 100
        
        # defining a box right in the middle of the screen
        start_box = (w//2 - roi_size//2, h//2 - roi_size//2, roi_size, roi_size)
        
        # starting the OpenCV tracker on that center box
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, start_box)
        self.initialized = True
        print("Tracker Initialized at center")

    def update(self, frame, frame_count):
        if frame is None: return frame
        
        if not self.initialized:
            # before we start, draw a guide so the user knows where to put the object
            h, w = frame.shape[:2]
            s = 100
            p1 = (w//2 - s//2, h//2 - s//2)
            p2 = (w//2 + s//2, h//2 + s//2)
            cv2.rectangle(frame, p1, p2, (200, 200, 200), 2)
            cv2.putText(frame, "Place object here & Wait", (p1[0]-20, p1[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            
            # wait 30 frames to give the camera time to settle, then lock on
            if frame_count > 30:
                self.init_tracker(frame)
            return frame

        # update the tracker to find the object in the new frame
        success, box = self.tracker.update(frame)
        if success:
            # if found, draw the box around it
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.bbox_color, 3)
            cv2.putText(frame, "Target Locked", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bbox_color, 2)
        else:
            # uh oh, lost it
            cv2.putText(frame, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
        return frame

# strategy 3: SAM2 Segmentation
# plays a pre-recorded video with masks because SAM2 is heavy
class SAM2Strategy(TrackerStrategy):
    def __init__(self):
        self.masks = []
        self.video_cap = None
        
        # need absolute paths so Flask doesn't get confused
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(base_dir, 'static', 'sam2_demo_video.mp4')
        self.mask_path = os.path.join(base_dir, 'static', 'segmentation.npz')
        
        print(f"DEBUG: SAM2 looking for video at: {self.video_path}")
        self.load_data()

    def load_data(self):
        # loading the pre-calculated segmentation masks
        if os.path.exists(self.mask_path):
            try:
                data = np.load(self.mask_path)
                if 'masks' in data:
                    self.masks = data['masks']
                print(f"Loaded {len(self.masks)} SAM2 masks.")
            except Exception as e:
                print(f"Error loading NPZ: {e}")
        else:
            print(f"ERROR: NPZ file not found at {self.mask_path}")
        
        # loading the video file
        if os.path.exists(self.video_path):
            self.video_cap = cv2.VideoCapture(self.video_path)
        else:
            print(f"ERROR: Video file not found at {self.video_path}")

    def update(self, frame, frame_count):
        # if the video isn't open, try to open it again
        if self.video_cap is None or not self.video_cap.isOpened():
            if os.path.exists(self.video_path):
                self.video_cap = cv2.VideoCapture(self.video_path)
            
            # if it still fails, tell the user something is wrong
            if self.video_cap is None or not self.video_cap.isOpened():
                if frame is not None:
                    cv2.putText(frame, "Video File Error - Check Terminal", (50,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    return frame
                return None

        # read the next frame from the video file
        ret, video_frame = self.video_cap.read()

        # if the video ends, loop back to the start
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, video_frame = self.video_cap.read()
            
        if not ret:
            return frame

        # apply the mask corresponding to the current frame
        if len(self.masks) > 0:
            idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) % len(self.masks)

            if idx < len(self.masks):
                # make sure mask size matches video frame size
                current_mask = self.masks[idx].astype(np.uint8)
                if current_mask.shape[:2] != video_frame.shape[:2]:
                    current_mask = cv2.resize(current_mask, (video_frame.shape[1], video_frame.shape[0]))

                # find contours on the mask and draw them on the video
                contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(video_frame, contours, -1, (255, 255, 0), 3)
            
            cv2.putText(video_frame, "SAM2", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return video_frame
        return video_frame