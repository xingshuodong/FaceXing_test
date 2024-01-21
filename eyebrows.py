import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the TensorFlow model
model_url = 'https://tfhub.dev/google/HRNet/ade20k-hrnetv2-w48/1'
model = hub.load(model_url)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open video file
cap = cv2.VideoCapture('testV.mp4')

def preprocess_for_model(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image / 255.0
    batch_image = np.expand_dims(normalized_image, axis=0)
    return batch_image

def calculate_bounding_box(hand_landmarks, image):
    x_max = max(int(lm.x * image.shape[1]) for lm in hand_landmarks.landmark)
    x_min = min(int(lm.x * image.shape[1]) for lm in hand_landmarks.landmark)
    y_max = max(int(lm.y * image.shape[0]) for lm in hand_landmarks.landmark)
    y_min = min(int(lm.y * image.shape[0]) for lm in hand_landmarks.landmark)
    width = x_max - x_min
    height = y_max - y_min
    width_expand = width * 0.5
    height_expand = height * 0.5
    x_min = max(x_min - width_expand, 0)
    x_max = min(x_max + width_expand, image.shape[1])
    y_min = max(y_min - height_expand, 0)
    y_max = min(y_max + height_expand, image.shape[0])
    return x_min, x_max, y_min, y_max

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
           x_min, x_max, y_min, y_max = calculate_bounding_box(hand_landmarks, image)
           x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
           x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
           y_min, y_max = max(0, y_min), min(image.shape[0], y_max)
           mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

           if cv2.waitKey(1) & 0xFF == ord('f'):
                cropped_image = image[y_min:y_max, x_min:x_max]
                cv2.imwrite('sample.jpg', cropped_image)
                processed_image = preprocess_for_model(cropped_image)
                predictions = model(processed_image)
                segmentation_map = np.argmax(predictions[0], axis=-1)
                max_label = np.max(segmentation_map)
                colors = np.random.randint(0, 255, size=(max_label + 1, 3))
                segmentation_image = colors[segmentation_map]
                segmentation_image_resized = cv2.resize(segmentation_image, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                overlay = cv2.addWeighted(image[y_min:y_max, x_min:x_max], 0.6, segmentation_image_resized, 0.4, 0)
                image[y_min:y_max, x_min:x_max] = overlay
                cv2.imshow('Segmentation Overlay', image)
                cv2.imwrite('segmentation_output.jpg', image)

    cv2.imshow('Hand Tracking with Segmentation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
