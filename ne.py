import mediapipe as mp
import serial
import time
import cv2
import math

arduino = serial.Serial('COM8', 9600, timeout=1)  
time.sleep(2)  

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def calculate_angle(p1, p2):
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    angle = math.degrees(math.atan2(-delta_y, delta_x))  
    angle = (angle + 360) % 360 
    return angle

cap = cv2.VideoCapture(0)

claw_angle = 90  
lift_angle = 90  
lift_speed = 2   
movement_distance = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

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
                    movement_distance = 5
                elif 80 <= angle <= 100:  
                    motor_command = f"MF{motor_speed}"
                    movement_distance = 10
                elif 170 <= angle <= 190:  
                    motor_command = f"ML{motor_speed}"
                    movement_distance = 5
                elif 260 <= angle <= 280: 
                    motor_command = f"MB{motor_speed}"
                    movement_distance = 7
                else:
                    movement_distance = 0
                    motor_command = "MS"
                
                arduino.write(f'{motor_command}\n'.encode())
                print(f"Motor Cmd: {motor_command}, Distance: {movement_distance} cm")

            print(f"Claw Angle: {claw_angle}, Lift Angle: {lift_angle}")

            cv2.putText(frame, f"Claw: {claw_angle} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Lift: {lift_angle} deg", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Cmd: {motor_command}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Move: {movement_distance} cm", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        arduino.write('MS\n'.encode())
        movement_distance = 0

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
