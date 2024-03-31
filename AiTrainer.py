import cv2
import numpy as np
import time
import PoseEstimationModule as pem

# For WebCam ___________________________________
# capWidth, capHeight = 1280, 720

# cap = cv2.VideoCapture(0)

# cap.set(3, capWidth)
# cap.set(4, capHeight)

cap = cv2.VideoCapture("dataset/3.mp4")
estimator = pem.poseEstimator()

count = 0
direction = 0  # 0 --> Up, 1 --> Down

ctime = 0
ptime = 0

while True:
    success, img = cap.read()
    frame = estimator.rescaleFrame(img, scale=0.4)
    frame = estimator.estimatePose(frame, draw=False)

    lmlist = estimator.getPosition(frame, draw=False)
    if len(lmlist) != 0:
        angle, frame = estimator.findAngle(frame, 11, 13, 15)  # Left Arm
        # angle, frame = estimator.findAngle(frame, 12, 14, 16)  # Right Arm

        # Checking for percentage
        per = np.interp(angle, (210, 290), (0, 100))


        # Check for curls
        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0

        # Dimensions of frame & Putting Information
        h, w, c = frame.shape
        cv2.rectangle(frame, (0, 3 * (h//4)), ((w//4), h), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f'{int(count)}', ((w//14), 11 * (h//12)), cv2.FONT_HERSHEY_PLAIN, h//120, (255, 0, 0), thickness=5)



    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime

    cv2.putText(frame, str(fps), (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), thickness=1)
    cv2.imshow("Video", frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
