# Import necessary libraries
import cv2                         # For video capture and image processing
import numpy as np                 # For numerical operations 
import PoseEstimationModule as pm  # Custom module for pose estimation (based on MediaPipe)
import time                        # For calculating FPS
import pyttsx3                     # Text-to-speech engine
import threading                   # To run speech in a separate thread (non-blocking)

# Function to speak without blocking main thread
def speak(text):
    threading.Thread(target=_speak_async, args=(text,), daemon=True).start()

# Helper function to run pyttsx3 in a new thread
def _speak_async(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speaking speed
    engine.say(text)
    engine.runAndWait()

# Initialize webcam capture
cap = cv2.VideoCapture(0)        # 0 is the default webcam
cap.set(3, 1280)  # Set video width
cap.set(4, 720)   # Set video height

# Variables for FPS calculation
pTime = 0

# Create pose detector object
detector = pm.poseDetector()

# Repetition counters and movement direction for both arms
count_right = 0
dir_right = 0  # 0 = down, 1 = up
last_announce_right = 0  # For speech only once every 10 reps

count_left = 0
dir_left = 0
last_announce_left = 0

# Start video loop
while True:
    success, img = cap.read()                 # Capture frame
    img = cv2.resize(img, (1280, 720))        # Resize for consistency
    img = cv2.flip(img, 1)                    # Mirror image for natural feel

    img = detector.findPose(img, draw=False)  # Detect pose
    lmList = detector.findPosition(img, draw=False)  # Get landmark list

    # If landmarks are detected
    if lmList:
        # --- RIGHT ARM CURL LOGIC ---
        angle_right = detector.findAngle(img, 11, 13, 15, draw=True)  # Shoulder-Elbow-Wrist angle
        per_right = np.interp(angle_right, (210, 300), (0, 100))      # Percentage for progress
        bar_right = np.interp(angle_right, (210, 300), (650, 100))    # Bar for visual feedback

        # Count logic based on angle percentage
        if per_right == 100 and dir_right == 0:
            count_right += 0.5
            dir_right = 1
        if per_right == 0 and dir_right == 1:
            count_right += 0.5
            dir_right = 0
            if int(count_right) % 10 == 0 and int(count_right) != last_announce_right:
                speak(f"You completed {int(count_right)} right arm curls!")
                last_announce_right = int(count_right)

        # Display right arm bar and count
        cv2.rectangle(img, (50, 450), (150, 720), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (50, int(bar_right)), (150, 720), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per_right)}%', (50, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'R:{int(count_right)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5)

        # --- LEFT ARM CURL LOGIC ---
        angle_left = detector.findAngle(img, 12, 14, 16, draw=True)  # Shoulder-Elbow-Wrist angle
        per_left = np.interp(angle_left, (210, 300), (0, 100))
        bar_left = np.interp(angle_left, (210, 300), (650, 100))

        if per_left == 100 and dir_left == 0:
            count_left += 0.5
            dir_left = 1
        if per_left == 0 and dir_left == 1:
            count_left += 0.5
            dir_left = 0
            if int(count_left) % 10 == 0 and int(count_left) != last_announce_left:
                speak(f"You completed {int(count_left)} left arm curls!")
                last_announce_left = int(count_left)

        # Display left arm bar and count
        cv2.rectangle(img, (1130, 450), (1230, 720), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (1130, int(bar_left)), (1230, 720), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per_left)}%', (1100, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'L:{int(count_left)}', (1050, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5)

        # --- BALANCE DETECTION LOGIC ---
        # Get shoulder and hip coordinates
        l_shoulder = lmList[11][1:]
        r_shoulder = lmList[12][1:]
        l_hip = lmList[23][1:]
        r_hip = lmList[24][1:]

        # Differences to check for imbalance
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        hip_diff = abs(l_hip[1] - r_hip[1])
        left_slope = abs(l_shoulder[0] - l_hip[0])
        right_slope = abs(r_shoulder[0] - r_hip[0])

        if shoulder_diff > 40 or hip_diff > 40 or abs(left_slope - right_slope) > 30:
            cv2.putText(img, "\u26a0\ufe0f Unbalanced Posture!", (400, 100),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            speak("Please keep your posture balanced")

    # --- FPS DISPLAY ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # --- SHOW OUTPUT WINDOW ---
    cv2.imshow("PoseFit Dual Arm with Balance", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
