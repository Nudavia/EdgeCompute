# -*-coding:utf-8-*-
import cv2

from models.face_detect.face_detect import FaceDetect
from models.expression_detect.expression_detect import ExpressionDetect
from models.text_detect.text_detect import TextDetect

# 网络模型  和  预训练模型
faceProto = "models/face_detect/opencv_face_detector.pbtxt"
faceModel = "models/face_detect/opencv_face_detector_uint8.pb"
bbox_net = cv2.dnn.readNet(faceModel, faceProto)


class Detector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.options = [FaceDetect(bbox_net), ExpressionDetect(bbox_net), TextDetect()]
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
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector()
    detector.run()
