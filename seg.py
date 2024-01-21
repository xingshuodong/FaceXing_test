import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

model_url = 'https://tfhub.dev/google/HRNet/ade20k-hrnetv2-w48/1'
model = hub.load(model_url)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils

LEFT_EYEBROW_INDICES: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
EYEBROW_INDICES = LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES

cap = cv2.VideoCapture('testV.mp4')

def is_holding_small_object(hand_landmarks, threshold=0.1):
    fingertip_indices = [4, 8, 12, 16, 20]
    palm_base_indices = [0, 5, 9, 13, 17]
    for fingertip_idx, palm_base_idx in zip(fingertip_indices, palm_base_indices):
        fingertip = hand_landmarks.landmark[fingertip_idx]
        palm_base = hand_landmarks.landmark[palm_base_idx]
        distance = ((fingertip.x - palm_base.x) ** 2 + (fingertip.y - palm_base.y) ** 2) ** 0.5
        if distance < threshold:
            return True
    return False

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

    # Expand the bounding box
    width_expand = width * 0.5
    height_expand = height * 0.5

    # Check which dimension to expand
    if width * 1.5 < height:
        # Expand width
        x_min = max(x_min - width_expand, 0)
        x_max = min(x_max + width_expand, image.shape[1])
    else:
        # Expand height
        y_min = max(y_min - height_expand, 0)
        y_max = min(y_max + height_expand, image.shape[0])

    return x_min, x_max, y_min, y_max




while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    face_results = face_mesh.process(image_rgb)
    if face_results.multi_face_landmarks:  # Check for face detection
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw only the eyebrow landmarks if a face is detected
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            for idx in EYEBROW_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_holding_small_object(hand_landmarks):
                x_min, x_max, y_min, y_max = calculate_bounding_box(hand_landmarks, image)
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

                if cv2.waitKey(1) & 0xFF == ord('f'):
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    cv2.imwrite('sample.jpg', cropped_image)
                    processed_image = preprocess_for_model(cropped_image)
                    predictions = model(processed_image)

                    # Process the predictions
                    segmentation_map = np.argmax(predictions[0], axis=-1)
                    max_label = np.max(segmentation_map)
                    np.random.seed(0)  # For reproducibility
                    colors = np.random.randint(0, 255, size=(max_label + 1, 3))
                    segmentation_image = colors[segmentation_map]
                    segmentation_image_resized = cv2.resize(segmentation_image, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                    segmentation_image_resized = segmentation_image_resized.astype(np.uint8)
                    bbox_image = image[y_min:y_max, x_min:x_max].astype(np.uint8)
                    overlay = cv2.addWeighted(bbox_image, 0.6, segmentation_image_resized, 0.4, 0)
                    image[y_min:y_max, x_min:x_max] = overlay
                    cv2.imshow('Segmentation Overlay', image)
                    cv2.imwrite('segmentation_output.jpg', image)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Hand Tracking with Segmentation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()