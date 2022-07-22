# -*-coding:utf-8-*-
import os

import cv2
import numpy as np

from models.face_detect.facenet import Facenet

# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class FaceDetect:
    def __init__(self, bbox_net):
        self.model = Facenet()
        self.std_imgs = []
        self.parent_dir = 'models/face_detect/face'
        self.labels = []
        for img in os.listdir(self.parent_dir):
            self.std_imgs.append(cv2.imread(os.path.join(self.parent_dir, img)))
            self.labels.append(img[:3])
        self.bbox_net = bbox_net

    def get_face(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
        frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.bbox_net.setInput(blob)
        detections = self.bbox_net.forward()  # 网络进行前向传播，检测人脸
        bbox = None
        max_confidence = conf_threshold
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_confidence:
                max_confidence = confidence
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bbox = [x1, y1, x2, y2]  # bounding box 的坐标
        if bbox is not None:
            bbox[0] = max(0, min(frame.shape[1], bbox[0]))
            bbox[1] = max(0, min(frame.shape[0], bbox[1]))
            bbox[2] = max(0, min(frame.shape[1], bbox[2]))
            bbox[3] = max(0, min(frame.shape[0], bbox[3]))
        return bbox

    def process(self, frame, keypressed):
        frame = cv2.flip(frame, 1)
        bbox = self.get_face(frame)
        print(bbox)
        if bbox is not None:
            min_distance = 1.0
            label = '？？？'
            for i, img in enumerate(self.std_imgs):
                distance = self.model.detect_cv_image(img, frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                if min_distance > distance:
                    min_distance = distance
                    label = self.labels[i]
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label + '(' + str(min_distance) + ')', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
        frame = cv2.flip(frame, 1)
        return frame
