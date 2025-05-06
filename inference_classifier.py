import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict.get('model', None)
    if model is None:
        print("Error: Model failed to load!")
        exit()
    else:
        print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1',
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'HELLO', 37: 'YES', 38: 'NO', 39: 'HOW ARE YOU'
}

engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  

last_detected_sign = None
tts_thread = None
last_speak_time = time.time()
last_hand_time = time.time()

root = tk.Tk()
root.title("Sign Language Translator")

video_label = tk.Label(root)
video_label.pack()

text_display = tk.Label(root, text="Recognized: ", font=("Arial", 20), fg="black")
text_display.pack()

fps_label = tk.Label(root, text="FPS: 0", font=("Arial", 12), fg="blue")
fps_label.pack()

prev_time = 0 

def speak(text):
    """Function to handle text-to-speech in a separate thread"""
    global tts_thread
    if tts_thread and tts_thread.is_alive():
        return  
    tts_thread = threading.Thread(target=lambda: engine.say(text) or engine.runAndWait())
    tts_thread.start()

def update_frame():
    global last_detected_sign, last_speak_time, last_hand_time, prev_time
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_label.config(text=f"FPS: {int(fps)}")

    if not results.multi_hand_landmarks:
        text_display.config(text="Recognized: No hand detected")
        frame_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_label.imgtk = frame_img
        video_label.configure(image=frame_img)
        root.after(10, update_frame)
        return

    last_hand_time = time.time()

    data_aux, x_, y_ = [], [], []

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 255, 0), -1) 

            x_.append(lm.x)
            y_.append(lm.y)

    if len(x_) == 0 or len(y_) == 0:
        root.after(10, update_frame)
        return

    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)

    for i in range(len(x_)):
        data_aux.append((x_[i] - min_x) / (max_x - min_x))
        data_aux.append((y_[i] - min_y) / (max_y - min_y))

    hand_width = max_x - min_x
    data_aux.append(hand_width)

    if len(data_aux) != 43:
        root.after(10, update_frame)
        return

    try:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
    except Exception as e:
        root.after(10, update_frame)
        return

    if predicted_character != last_detected_sign:
        last_detected_sign = predicted_character
        last_speak_time = time.time()

    if time.time() - last_speak_time > 3 and last_detected_sign:
        speak(last_detected_sign)
        last_detected_sign = None  

    text_display.config(text=f"Recognized: {last_detected_sign if last_detected_sign else '...' }")

    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
    x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    frame_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.imgtk = frame_img
    video_label.configure(image=frame_img)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()