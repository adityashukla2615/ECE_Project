import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import serial
import math

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


try:
    arduino = serial.Serial('COM8', 9600, timeout=1) 
    time.sleep(2)  
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)


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


claw_angle = 90  
lift_angle = 90   
lift_speed = 2   
claw_min = 0      
claw_max = 180   


last_detected_sign = None
tts_thread = None
last_speak_time = time.time()
last_hand_time = time.time()
current_direction = "STOP"  
current_claw_status = "HOLD" 

root = tk.Tk()
root.title("Sign Language Translator")

video_label = tk.Label(root)
video_label.pack()

text_display = tk.Label(root, text="Recognized: ", font=("Arial", 20), fg="black")
text_display.pack()


direction_display = tk.Label(root, text="Direction: STOP", font=("Arial", 16), fg="blue")
direction_display.pack()


claw_display = tk.Label(root, text="Claw: HOLD", font=("Arial", 16), fg="purple")
claw_display.pack()


fps_label = tk.Label(root, text="FPS: 0", font=("Arial", 12), fg="green")
fps_label.pack()

prev_time = 0  

def speak(text):
    """Function to handle text-to-speech in a separate thread"""
    global tts_thread
    if tts_thread and tts_thread.is_alive():
        return  
    tts_thread = threading.Thread(target=lambda: engine.say(text) or engine.runAndWait())
    tts_thread.start()

def get_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def calculate_angle(p1, p2):
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    angle = math.degrees(math.atan2(-delta_y, delta_x)) 
    angle = (angle + 360) % 360  
    return angle

def send_motor_command(direction):
    """Send motor command and update GUI"""
    global current_direction
    if direction != current_direction:
        current_direction = direction
        direction_display.config(text=f"Direction: {direction}")
        if arduino:
            arduino.write(f'{direction}\n'.encode())

def send_claw_command(angle, status):
    """Send claw command and update GUI"""
    global current_claw_status
    angle = int(angle)
    if status != current_claw_status:
        current_claw_status = status
        claw_display.config(text=f"Claw: {status}")
    if arduino:
        arduino.write(f'C{angle}\n'.encode())

def update_frame():
    global last_detected_sign, last_speak_time, last_hand_time, prev_time, lift_angle, current_direction, claw_angle, current_claw_status
    
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

    
    right_hand_detected = False
    left_hand_detected = False
    fingers_joined = False

    
    if not results.multi_hand_landmarks:
        text_display.config(text="Recognized: No hand detected")
        if time.time() - last_hand_time > 0.5 and current_direction != "STOP":
            send_motor_command("MS")  # Stop command
        frame_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_label.imgtk = frame_img
        video_label.configure(image=frame_img)
        root.after(10, update_frame)
        return

    last_hand_time = time.time()  

    data_aux, x_, y_ = [], [], []

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        mp_drawing.draw_landmarks(
            frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

       
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        little_tip = hand_landmarks.landmark[20]

       
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 255, 0), -1)  
            x_.append(lm.x)
            y_.append(lm.y)

        hand_label = handedness.classification[0].label  
        
        if hand_label == 'Left':  
            left_hand_detected = True
            thumb_index_dist = get_distance(thumb_tip, index_tip)
            
            min_dist = 0.02  
            max_dist = 0.2    
            
            
            norm_dist = max(0, min(1, (thumb_index_dist - min_dist) / (max_dist - min_dist)))
            
            
            claw_angle = claw_max - int(norm_dist * (claw_max - claw_min))
            
            
            if norm_dist < 0.1:
                claw_status = "CLOSED"
            elif norm_dist > 0.9:
                claw_status = "OPEN"
            else:
                claw_status = "HOLD"
            
            send_claw_command(claw_angle, claw_status)

        
            if ring_tip.y < index_tip.y:  
                lift_angle = min(lift_angle + lift_speed, 180)
                if arduino:
                    arduino.write(f'H{lift_angle}\n'.encode())
            elif little_tip.y < index_tip.y:  
                lift_angle = max(lift_angle - lift_speed, 0)
                if arduino:
                    arduino.write(f'H{lift_angle}\n'.encode())

        elif hand_label == 'Right':  
            right_hand_detected = True
            
            fingers_joined = get_distance(index_tip, middle_tip) < 0.05
            
            if fingers_joined:
                angle = calculate_angle(index_tip, ring_tip)
                motor_speed = 80  

                if 350 <= angle or angle <= 10:  # Right
                    send_motor_command(f"MR{motor_speed}")
                elif 80 <= angle <= 100:  # Forward
                    send_motor_command(f"MF{motor_speed}")
                elif 170 <= angle <= 190:  # Left
                    send_motor_command(f"ML{motor_speed}")
                elif 260 <= angle <= 280:  # Backward
                    send_motor_command(f"MB{motor_speed}")
                else:
                    send_motor_command("MS")
            else:
                send_motor_command("MS")

    
    if right_hand_detected and not fingers_joined and current_direction != "STOP":
        send_motor_command("MS")

    
    if not left_hand_detected and current_claw_status != "HOLD":
        current_claw_status = "HOLD"
        claw_display.config(text="Claw: HOLD")

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

    
    text_display.config(text=f"Recognized: {last_detected_sign if last_detected_sign else '...'}")

    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
    x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.putText(frame_rgb, f"Car: {current_direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_rgb, f"Claw: {current_claw_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    frame_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.imgtk = frame_img
    video_label.configure(image=frame_img)

    root.after(10, update_frame)


update_frame()
root.mainloop()


cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()