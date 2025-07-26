# squat_counter_threaded.py

import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import threading

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Pose detection setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables
count = 0
direction = 0
last_announce = 0
last_balance_alert = 0
balance_alert_interval = 5  # seconds
pTime = 0

def find_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    lmList = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            lmList.append([id, int(lm.x * w), int(lm.y * h)])

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if len(lmList) > 27:
            # Left leg for squat (hip-knee-ankle)
            hip = lmList[23][1:]
            knee = lmList[25][1:]
            ankle = lmList[27][1:]

            angle = find_angle(hip, knee, ankle)
            per = np.interp(angle, (60, 170), (100, 0))
            bar = np.interp(angle, (60, 170), (650, 100))

            # Counter logic
            if per == 100:
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0
                    if int(count) % 10 == 0 and int(count) != last_announce:
                        speak_async(f"You have completed {int(count)} squats")
                        last_announce = int(count)

            # Draw bar and counter
            cv2.rectangle(img, (50, 100), (150, 650), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(img, (50, int(bar)), (150, 650), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Squats: {int(count)}', (50, 700), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

            # Posture balance check
            l_shoulder = lmList[11][1:]
            r_shoulder = lmList[12][1:]
            l_hip = lmList[23][1:]
            r_hip = lmList[24][1:]

            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
            hip_diff = abs(l_hip[1] - r_hip[1])
            left_slope = abs(l_shoulder[0] - l_hip[0])
            right_slope = abs(r_shoulder[0] - r_hip[0])
            current_time = time.time()

            if shoulder_diff > 40 or hip_diff > 40 or abs(left_slope - right_slope) > 30:
                cv2.putText(img, "\u26a0\ufe0f Unbalanced Posture!", (400, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                if current_time - last_balance_alert > balance_alert_interval:
                    speak_async("Please keep your posture balanced")
                    last_balance_alert = current_time

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Squat Counter with Balance Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
