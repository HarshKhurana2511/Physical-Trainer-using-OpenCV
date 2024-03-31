import cv2
import mediapipe as mp
import time
import math

class poseEstimator():
    def __init__(self, mode=False, modelComplex=1, smoothLmks=True, enableSegment=False, smoothSegment = True, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.modelComplex = modelComplex
        self.smoothLmks = smoothLmks
        self.enableSegment = enableSegment
        self.smoothSegment = smoothSegment
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def rescaleFrame(self, frame, scale=0.5):
        # Works for Images, Videos and Live Videos
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def estimatePose(self, frame, draw=True):
        self.results = self.pose.process(frame)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame

    def getPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, frame, p1, p2, p3, draw=True):

        # Getting Landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2,) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 255), 3)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(frame, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(frame, str(int(angle)), (x2-50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle, frame

def main():
    ctime = 0
    ptime = 0

    cap = cv2.VideoCapture('pose_data/1.mp4')

    estimator = poseEstimator()

    while True:
        success, img = cap.read()
        frame = estimator.rescaleFrame(img, scale=0.2)
        frame = estimator.estimatePose(frame)

        ctime = time.time()
        fps = int(1 / (ctime - ptime))
        ptime = ctime

        cv2.putText(frame, str(fps), (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), thickness=1)
        cv2.imshow("Video", frame)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()