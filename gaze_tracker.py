import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time

SMOOTHING_FRAMES = 10
SIDE_LOOK_TIME = 3
PHONE_LOOK_TIME = 2
ABSENCE_TIME = 2

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33,133]
RIGHT_EYE = [362,263]

LEFT_IRIS = [468,469,470,471]
RIGHT_IRIS = [473,474,475,476]

class GazeTracker:

    def __init__(self):

        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=5,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.gaze_history = deque(maxlen=SMOOTHING_FRAMES)

        self.last_face_time = time.time()
        self.side_start_time = None
        self.phone_start_time = None
        self.center_y = None


    def _coord(self, lm, i, w, h):
        p = lm[i]
        return np.array([p.x*w, p.y*h])


    def _iris_center(self, lm, ids, w, h):
        pts = [self._coord(lm,i,w,h) for i in ids]
        return np.mean(pts,axis=0)


    def _eye_center(self,lm,ids,w,h):
        pts=[self._coord(lm,i,w,h) for i in ids]
        return np.mean(pts,axis=0)


    def compute_gaze(self,lm,w,h):

        left_eye=self._eye_center(lm,LEFT_EYE,w,h)
        right_eye=self._eye_center(lm,RIGHT_EYE,w,h)

        left_iris=self._iris_center(lm,LEFT_IRIS,w,h)
        right_iris=self._iris_center(lm,RIGHT_IRIS,w,h)

        lw=np.linalg.norm(self._coord(lm,33,w,h)-self._coord(lm,133,w,h))
        rw=np.linalg.norm(self._coord(lm,362,w,h)-self._coord(lm,263,w,h))

        gx=((left_iris[0]-left_eye[0])/lw + (right_iris[0]-right_eye[0])/rw)/2
        gy=((left_iris[1]-left_eye[1])/lw + (right_iris[1]-right_eye[1])/rw)/2

        return gx,gy


    def head_yaw(self,lm,w,h):

        nose=lm[1].x*w
        left=lm[234].x*w
        right=lm[454].x*w

        center=(left+right)/2
        yaw=(nose-center)/(right-left)

        return yaw*60


    def smooth(self,direction):

        self.gaze_history.append(direction)

        if len(self.gaze_history)<3:
            return direction

        return Counter(self.gaze_history).most_common(1)[0][0]


    def classify(self,gx,gy,yaw):

        if abs(gx)<0.15 and abs(gy)<0.08 and abs(yaw)<10:
            return "center"

        if gx<-0.28 or yaw<-20:
            return "left"

        if gx>0.28 or yaw>20:
            return "right"

        if gy<-0.18:
            return "up"

        if gy>0.18:
            return "down"

        return "center"


    def check_violations(self,direction,yaw):

        violations=[]
        now=time.time()

        if direction in ["left","right"]:
            if self.side_start_time is None:
                self.side_start_time=now
            elif now-self.side_start_time> SIDE_LOOK_TIME:
                violations.append("second_monitor")
        else:
            self.side_start_time=None

        if direction=="down":
            if self.phone_start_time is None:
                self.phone_start_time=now
            elif now-self.phone_start_time> PHONE_LOOK_TIME:
                violations.append("phone_usage")
        else:
            self.phone_start_time=None

        return violations


    def process_frame(self,frame):

        start=time.time()

        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=self.face_mesh.process(rgb)

        data={
            "gaze":None,
            "violations":[],
            "faces":0,
            "cpu_ms":None
        }

        if not results.multi_face_landmarks:

            if time.time()-self.last_face_time>ABSENCE_TIME:
                data["violations"]=["eye_absence"]

            data["faces"]=0
            return data,frame


        faces=len(results.multi_face_landmarks)
        data["faces"]=faces

        if faces>1:
            data["violations"]=["multiple_faces"]
            return data,frame

        self.last_face_time=time.time()

        lm=results.multi_face_landmarks[0].landmark

        gx,gy=self.compute_gaze(lm,w,h)

        if self.center_y is None:
            self.center_y=gy

        gy=gy-self.center_y

        yaw=self.head_yaw(lm,w,h)

        direction=self.classify(gx,gy,yaw)

        direction=self.smooth(direction)

        violations=self.check_violations(direction,yaw)

        data["gaze"]=direction
        data["violations"]=violations

        cv2.putText(frame,f"Gaze:{direction}",(10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        end=time.time()

        data["cpu_ms"]=(end-start)*1000

        return data,frame