import mediapipe as mp
import serial
import time
import cv2
import math
import pickle
import numpy as np
import pyttsx3
import threading
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

arduino = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1',
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'HELLO', 37: 'YES', 38: 'NO', 39: 'HOW ARE YOU'
}

root = tk.Tk()
root.title("Sign Language & Robot Control")
video_label = tk.Label(root)
video_label.pack()
text_display = tk.Label(root, text="Recognized: ", font=("Arial", 20), fg="black")
text_display.pack()
fps_label = tk.Label(root, text="FPS: 0", font=("Arial", 12), fg="blue")
fps_label.pack()

def get_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def calculate_angle(p1, p2):
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    angle = math.degrees(math.atan2(-delta_y, delta_x))
    angle = (angle + 360) % 360
    return angle

cap = cv2.VideoCapture(0)
prev_time = 0

claw_angle = 90
lift_angle = 90
lift_speed = 2
movement_distance = 0
last_detected_sign = None
last_speak_time = time.time()

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

def update_frame():
    global last_detected_sign, last_speak_time, prev_time
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

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            little_tip = hand_landmarks.landmark[20]
            hand_label = handedness.classification[0].label

            if hand_label == 'Left':
                claw_distance = get_distance(thumb_tip, index_tip)
                claw_angle = int(min(max((claw_distance * 800), 0), 180))
                arduino.write(f'C{claw_angle}\n'.encode())
                
                if middle_tip.y < index_tip.y and little_tip.y > index_tip.y:
                    lift_angle = max(lift_angle - lift_speed, 0)
                elif little_tip.y < index_tip.y and middle_tip.y > index_tip.y:
                    lift_angle = min(lift_angle + lift_speed, 180)
                
                arduino.write(f'H{lift_angle}\n'.encode())

            elif hand_label == 'Right':
                angle = calculate_angle(index_tip, ring_tip)
                motor_command = "MS"
                motor_speed = 80

                if 350 <= angle or angle <= 10:
                    motor_command = f"MR{motor_speed}"
                elif 80 <= angle <= 100:
                    motor_command = f"MF{motor_speed}"
                elif 170 <= angle <= 190:
                    motor_command = f"ML{motor_speed}"
                elif 260 <= angle <= 280:
                    motor_command = f"MB{motor_speed}"
                
                arduino.write(f'{motor_command}\n'.encode())
                text_display.config(text=f"Recognized: {motor_command}")
    
    frame_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.imgtk = frame_img
    video_label.configure(image=frame_img)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
arduino.close()