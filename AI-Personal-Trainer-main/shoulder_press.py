# shoulder_press_counter.py

import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import threading

# Voice setup
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

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

count = 0
direction = 0
last_announce = 0
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

        if len(lmList) > 15:
            # Get positions
            r_elbow = lmList[14][1:]
            r_wrist = lmList[16][1:]
            r_shoulder = lmList[12][1:]

            # Calculate vertical distance from wrist/elbow to shoulder
            wrist_to_shoulder_diff = r_shoulder[1] - r_wrist[1]
            elbow_to_shoulder_diff = r_shoulder[1] - r_elbow[1]

            # Normalize for bar and percentage (adjust thresholds as needed)
            max_diff = 200
            min_diff = 20
            wrist_to_shoulder_diff = np.clip(wrist_to_shoulder_diff, min_diff, max_diff)
            per = np.interp(wrist_to_shoulder_diff, (min_diff, max_diff), (0, 100))
            bar = np.interp(wrist_to_shoulder_diff, (min_diff, max_diff), (650, 150))

            # Draw bar
            cv2.rectangle(img, (50, 150), (150, 650), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(bar)), (150, 650), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (60, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # Count reps based on movement direction
            if per == 100:
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                if direction == 1:
                    count += 0.5
                    direction = 0
                    if int(count) != last_announce and int(count) % 10 == 0:
                        speak_async(f"You have completed {int(count)} shoulder presses")
                        last_announce = int(count)

            # Display count
            cv2.putText(img, f'Shoulder Press: {int(count)}', (50, 700),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Shoulder Press Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
