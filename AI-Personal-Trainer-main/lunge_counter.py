# lunges_counter.py

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time

# Setup TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_async(text):
    threading.Thread(target=lambda: speak(text), daemon=True).start()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam config
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Counters and direction trackers
left_count = 0
right_count = 0
left_dir = 0
right_dir = 0
pTime = 0
last_announce = 0

def calculate_angle(a, b, c):
    a = np.array(a)  # hip
    b = np.array(b)  # knee
    c = np.array(c)  # ankle
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
        h, w, _ = img.shape
        for id, lm in enumerate(results.pose_landmarks.landmark):
            lmList.append([id, int(lm.x * w), int(lm.y * h)])
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if len(lmList) >= 29:
            # LEFT LEG
            l_hip = lmList[23][1:]
            l_knee = lmList[25][1:]
            l_ankle = lmList[27][1:]
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            l_per = np.interp(l_angle, (60, 170), (100, 0))  # Down = 100, Up = 0

            if l_per == 100 and left_dir == 0:
                left_dir = 1
            if l_per == 0 and left_dir == 1:
                left_count += 1
                left_dir = 0
                speak_async(f"Left leg lunge {left_count}")

            # RIGHT LEG
            r_hip = lmList[24][1:]
            r_knee = lmList[26][1:]
            r_ankle = lmList[28][1:]
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)
            r_per = np.interp(r_angle, (60, 170), (100, 0))  # Down = 100, Up = 0

            if r_per == 100 and right_dir == 0:
                right_dir = 1
            if r_per == 0 and right_dir == 1:
                right_count += 1
                right_dir = 0
                speak_async(f"Right leg lunge {right_count}")

            # Draw angle + bars
            cv2.putText(img, f'L: {int(l_angle)} deg', (30, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.putText(img, f'R: {int(r_angle)} deg', (30, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            cv2.putText(img, f'L Lunges: {left_count}', (30, 650), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)
            cv2.putText(img, f'R Lunges: {right_count}', (30, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Lunges Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
