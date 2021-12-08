from __future__ import division
import os
import cv2
import dlib
import matplotlib.pyplot as plt
from .eye import Eye
from .calibration import Calibration

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self,videopath):
        self.frame = None
        self.d = None
        self.landmarks = None

        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # required data #
        self.eyebrow_height = []
        self.eyebrow_pitch = []
        self.mouth_height = []
        self.mouth_pitch = []
        self.eyes_pitch=[]
        self.outblinking = 0
        self.outright = 0
        self.outleft = 0
        self.outcenter = 0

        self.videopath = videopath

        # _face_detector is used to detect faces
        self.detector = dlib.get_frontal_face_detector()
        self.cap = cv2.VideoCapture(self.videopath)

        # predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self.predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def learning_face(self):

        while(self.cap.isOpened()):
            flag, im_rd = self.cap.read()

            if(flag==False):
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.frame = im_rd
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(frame,0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if(len(faces)!=0):

                for k, self.d in enumerate(faces):
                    self.landmarks = self.predictor(frame, faces[0])
                    im_rd = self.annotated_frame()

                    stand = self.landmarks.part(9).x - self.landmarks.part(7).x

                    # start 眉毛活動/緊皺 #
                    eb_height = 0
                    eb_pitch = 0
                    for j in range(17,26):
                        eb_height += (self.landmarks.part(8).y - self.landmarks.part(j).y)/stand
                    eb_pitch += (self.landmarks.part(22).x - self.landmarks.part(21).x)/stand

                    self.eyebrow_height.append(eb_height/10)
                    self.eyebrow_pitch.append(eb_pitch)
                    # end 眉毛活動/緊皺 #

                    # start 嘴角上揚/微張 #
                    ms_height = 0
                    ms_pitch = 0
                    ms_height += (self.landmarks.part(8).y - self.landmarks.part(48).y)/stand
                    ms_height += (self.landmarks.part(8).y - self.landmarks.part(54).y)/stand
                    ms_pitch += (self.landmarks.part(54).x - self.landmarks.part(48).x)/stand

                    self.mouth_height.append(ms_height/2)
                    self.mouth_pitch.append(ms_pitch)
                    # end 嘴角上揚/微張 #
                        
                    self.eye_left = Eye(frame, self.landmarks, 0, self.calibration)
                    self.eye_right = Eye(frame, self.landmarks, 1, self.calibration)
                        
                    # start 眼睛眨眼 #   
                    self.eyes_pitch.append((self.landmarks.part(41).y - self.landmarks.part(37).y)/stand)
                    # end 眼睛眨眼 #

                    if self.is_right():
                        self.outright+=1
                        
                    elif self.is_left():
                        self.outleft+=1
                        
                    elif self.is_center():
                        self.outcenter+=1
                        
            else:
                cv2.putText(im_rd, "No Face or too many Faces", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                
            if (cv2.waitKey(1) == ord('q')):
                break
            
            # cv2.imshow("Demo", im_rd)

        self.cap.release()

        cv2.destroyAllWindows()

        return self.eyebrow_height, self.eyebrow_pitch, self.mouth_height, self.mouth_pitch, self.eyes_pitch, self.outright, self.outleft, self.outcenter

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.45

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 6

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()

            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        for i in range(68):
            cv2.circle(frame, (self.landmarks.part(i).x, self.landmarks.part(i).y), 2, (0, 255, 0), -1, 8)
                        
        return frame