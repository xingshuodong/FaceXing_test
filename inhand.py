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

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

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

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks.
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the bounding box of the hand
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            # Draw the bounding box
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Show the image.
    cv2.imshow('MediaPipe Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
        break

cap.release()
cv2.destroyAllWindows()
