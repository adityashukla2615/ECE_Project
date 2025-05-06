import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

   
    dot = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    angle = np.arccos(dot / (mag_v1 * mag_v2))
    return np.degrees(angle)

def normalize_landmarks(landmarks):
    x_vals = [landmark.x for landmark in landmarks]
    y_vals = [landmark.y for landmark in landmarks]

    min_x = min(x_vals)
    min_y = min(y_vals)
    max_x = max(x_vals)
    max_y = max(y_vals)

    normalized = []
    for landmark in landmarks:
        normalized.append((landmark.x - min_x) / (max_x - min_x))
        normalized.append((landmark.y - min_y) / (max_y - min_y))
    
    return normalized

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.isdir(dir_path):
        continue  
    
    for img_path in os.listdir(dir_path):
        img_file_path = os.path.join(dir_path, img_path)
        
        if not img_file_path.lower().endswith(('png', 'jpg', 'jpeg')): 
            continue
        
        data_aux = []

        img = cv2.imread(img_file_path)
        if img is None:
            print(f"Warning: Could not read image {img_file_path}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) == 21: 
                    normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                    data_aux.extend(normalized_landmarks)

                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

                    angle = calculate_angle((wrist.x, wrist.y), (index_finger.x, index_finger.y), (thumb.x, thumb.y))
                    data_aux.append(angle)

                else:
                    print(f"Warning: Missing landmarks in image {img_file_path}. Skipping.")
                    continue

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)

        else:
            print(f"Warning: No hand landmarks detected in image {img_file_path}. Skipping.")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete and saved to 'data.pickle'")
