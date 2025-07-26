import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import threading

# Voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

count = 0
direction = 0
last_announce = 0
last_balance_alert = 0
balance_alert_interval = 5
pTime = 0

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

        if len(lmList) > 28:
            l_hand = lmList[15][1:]
            r_hand = lmList[16][1:]
            l_foot = lmList[27][1:]
            r_foot = lmList[28][1:]
            shoulder = lmList[12][1:]

            hands_up = l_hand[1] < shoulder[1] and r_hand[1] < shoulder[1]
            feet_apart = abs(l_foot[0] - r_foot[0]) > 400

            # Percentage bar logic
            max_foot_distance = 500  # Max distance when feet are far apart
            min_foot_distance = 150  # Min distance when feet are close together
            foot_distance = abs(l_foot[0] - r_foot[0])
            per = np.interp(foot_distance, [min_foot_distance, max_foot_distance], [0, 100])
            bar = np.interp(foot_distance, [min_foot_distance, max_foot_distance], [650, 100])

            # Jumping jacks count logic
            if hands_up and feet_apart:
                if direction == 0:
                    count += 0.5
                    direction = 1
            elif not hands_up and not feet_apart:
                if direction == 1:
                    count += 0.5
                    direction = 0
                    if int(count) % 10 == 0 and int(count) != last_announce:
                        speak_async(f"You have completed {int(count)} jumping jacks")
                        last_announce = int(count)

            # Draw the percentage bar
            cv2.rectangle(img, (50, 100), (150, 650), (0, 255, 0), 2)
            cv2.rectangle(img, (50, int(bar)), (150, 650), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (60, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.putText(img, f'Jumping Jacks: {int(count)}', (200, 700), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

            # Balance detection
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
                cv2.putText(img, "\u26a0 Unbalanced Posture!", (400, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                if current_time - last_balance_alert > balance_alert_interval:
                    speak_async("Please keep your posture balanced")
                    last_balance_alert = current_time

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Jumping Jacks Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
