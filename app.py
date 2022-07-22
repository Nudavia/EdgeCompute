# -*-coding:utf-8-*-
import socket
import cv2
from models.mask.masks import Mask
from models.hand.hands import Hand
from models.pose.pose import Pose
from models.segement.segment import Segment
from models.face_mesh.face_mesh import FaceMesh
from models.objectron.objectron import Objectron


class ALL:
    def __init__(self):
        self.options = [Hand(), Pose(), FaceMesh()]

    def process(self, frame, keypressed):
        output = frame
        for opt in self.options:
            output = opt.process(output, keypressed)
            if output is None:
                return None
        return output


class Detector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.options = [Segment(), Hand(), Pose(), FaceMesh(), Mask(), ALL()]
        self.index = 0
        self.func = self.options[self.index]
        self.keypressed = 255

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            else:
                output = self.func.process(frame, self.keypressed)
                output = cv2.flip(output, 1)
                if output is None:
                    output = cv2.flip(frame, 1)
                if self.keypressed == 27:
                    break
                elif self.keypressed == ord('a'):
                    self.index = (self.index - 1) % len(self.options)
                    self.func = self.options[self.index]
                elif self.keypressed == ord('d'):
                    self.index = (self.index + 1) % len(self.options)
                    self.func = self.options[self.index]
                cv2.imshow('', output)
                self.keypressed = cv2.waitKey(1)
                print(self.keypressed)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector()
    detector.run()
