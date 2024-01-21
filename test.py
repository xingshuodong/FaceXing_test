import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import subprocess

# Initialize MediaPipe solutions.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Indices for the right eye landmarks based on the Face Mesh diagram.
RIGHT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160,
    159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33
]

# Capture video from the webcam.
cap = cv2.VideoCapture('testV.mp4')

# Path for the CSV file
csv_file_path = "landmarks.csv"

# Open the CSV file for appending
with open(csv_file_path, mode="a", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write header if file is empty
    if os.stat(csv_file_path).st_size == 0:
        headers = ['Timestamp', 'EyeLandmarkX', 'EyeLandmarkY', 'HandLandmarkX', 'HandLandmarkY']
        csv_writer.writerow(headers)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Convert the image color from BGR to RGB and process it with MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh and hands
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)

        # Draw the face mesh and hand landmarks on the image.
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        timestamp = time.time()
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw all face landmarks.
                mp_draw.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
                # Draw right eye landmarks with a different color.
                for idx in RIGHT_EYE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 1, (0, 250, 0), -1)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks.
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )

        # Show the image.
        cv2.imshow('MediaPipe Face and Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
            break

cap.release()
cv2.destroyAllWindows()

# Open the CSV file after the script ends
if os.name == 'nt':  # for Windows
    os.startfile(csv_file_path)
else:  # for macOS and Linux
    subprocess.call(['open', csv_file_path])
