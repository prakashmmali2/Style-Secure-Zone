import os
import cv2
import cvzone
import mediapipe as mp

# Initialize webcam or video stream
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Unable to open the webcam or video stream at index {camera_index}.")
    exit()

# MediaPipe Pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Paths for resources
shirtFolderPath = r"Resource\shirt"

# Load shirts
listShirts = [f for f in os.listdir(shirtFolderPath) if f.endswith('.png')]
preloadedShirts = []
for shirt in listShirts:
    imgShirt = cv2.imread(os.path.join(shirtFolderPath, shirt), cv2.IMREAD_UNCHANGED)
    if imgShirt is not None:
        if imgShirt.shape[2] == 3:
            imgShirt = cv2.cvtColor(imgShirt, cv2.COLOR_BGR2BGRA)
            imgShirt[:, :, 3] = 255
        preloadedShirts.append(imgShirt)

# Check if shirts are loaded
if not preloadedShirts:
    print("Error: No shirts found in the specified directory.")
    exit()

# Variables for interaction
imageNumber = 0
shirtRatioHeightWidth = 581 / 440  # Adjust based on the shirt image dimensions

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read the video frame or end of video.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        lm11 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm12 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate shoulder distance
        shoulder_distance = abs(lm11.x - lm12.x) * img.shape[1]

        # Ensure imageNumber is within the bounds of the preloadedShirts list
        if imageNumber >= len(preloadedShirts):
            imageNumber = len(preloadedShirts) - 1  # Reset to last valid index if out of range

        # Resize shirt dynamically
        imgShirt = preloadedShirts[imageNumber]
        shirt_width = int(shoulder_distance * 1.2)  # Adjust multiplier for better fit
        shirt_height = int(shirt_width * shirtRatioHeightWidth)
        imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))

        # Calculate shirt position
        shirt_x = int(lm12.x * img.shape[1]) - shirt_width // 2
        shirt_y = int(lm12.y * img.shape[0]) - shirt_height // 3

        # Overlay the shirt
        img = cvzone.overlayPNG(img, imgShirt, (shirt_x, shirt_y))

    # Display the result
    cv2.imshow("Image", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()