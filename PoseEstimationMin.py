import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('pose_data/1.mp4')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def rescaleFrame(frame, scale=0.5):
    # Works for Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

ctime = 0
ptime = 0

while True:
    success, img = cap.read()
    frame = rescaleFrame(img, scale=0.2)
    results = pose.process(frame)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            print(id, landmark)
            h, w, c = img.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            print(f"Id: {id}\nX: {cx}\nY: {cy}")

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime

    cv2.putText(frame, str(fps), (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), thickness=1)
    cv2.imshow("Video", frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()