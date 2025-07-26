# Import required libraries
import cv2                      # For image/video processing
import mediapipe as mp          # For pose detection
import math                     # For calculating angles between joints
import time                     # For FPS measurement (used in main)

# Define the pose detector class
class poseDetector():
    def __init__(self):
        # Initialize MediaPipe Pose solution
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()  # Load pose model
        self.mpDraw = mp.solutions.drawing_utils  # For drawing landmarks on image

    def findPose(self, img, draw=True):
        # Convert image from BGR to RGB (MediaPipe uses RGB)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and detect pose
        self.results = self.pose.process(imgRGB)

        # If landmarks are found, optionally draw them
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS  # Draw connections between joints
                )
        return img  # Return image with/without drawing

    def findPosition(self, img, draw=True):
        self.lmList = []  # List to hold landmark positions (id, x, y)

        # Check if pose landmarks are detected
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # Get image dimensions

                # Convert normalized coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Save the id and position in the list
                self.lmList.append([id, cx, cy])

                # Optionally draw small circle on each joint
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList  # Return list of landmarks (joint positions)

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get (x, y) coordinates of the three points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle using the arctangent difference formula
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))

        # Make sure angle is positive
        if angle < 0:
            angle += 360

        # Optionally draw lines and angle on the image
        if draw:
            # Draw lines between the joints
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)

            # Draw circles on joints
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            # Display the angle value on the image
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle  # Return the calculated angle



