# pushup_counter_fullbody_mediapipe.py

import cv2
import numpy as np
import time
import threading
import pyttsx3
import mediapipe as mp

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()

# Pose detection setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
count = 0
direction = 0
last_announce = 0
last_balance_alert = 0
balance_alert_interval = 5  # seconds

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    lmList = []
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

    if lmList:
        # Push-up logic using right shoulder (11), right elbow (13), right wrist (15)
        a = np.array(lmList[11][1:])
        b = np.array(lmList[13][1:])
        c = np.array(lmList[15][1:])

        # Calculate angle at elbow
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle

        # Convert angle to percentage and bar
        per = np.interp(angle, (90, 160), (100, 0))
        bar = np.interp(angle, (90, 160), (100, 650))

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
                    speak(f"You have completed {int(count)} pushups")
                    last_announce = int(count)

        # Draw bar and count
        cv2.rectangle(img, (50, 100), (150, 650), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (50, int(bar)), (150, 650), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'Pushups: {int(count)}', (50, 700), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

        # Balance check using shoulders and hips
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
                speak("Please keep your posture balanced")
                last_balance_alert = current_time

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Push-up Counter with Balance Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()