
import cv2
import mediapipe as mp
import csv
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

csv_file = open('pose_data.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'nose_x', 'nose_y', 'right_wrist_x', 'right_wrist_y'])

cap = cv2.VideoCapture(0)

print("Starting Pose Tracking Press 'q' to quit.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image_rgb)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        try:
            row = [time.time()]

            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                row.append(nose.x)
                row.append(nose.y)
            else:
                row.extend([0, 0])

            if results.right_hand_landmarks:
                wrist = results.right_hand_landmarks.landmark[0]
                row.append(wrist.x)
                row.append(wrist.y)
            else:
                row.extend([0, 0])
                
            writer.writerow(row)

        except Exception as e:
            pass

        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()