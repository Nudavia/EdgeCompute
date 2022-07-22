# -*-coding:utf-8-*-

import argparse
import warnings

import cv2
import torchvision.transforms as transforms
from PIL import Image
from models import *
import torch

from models.expression_detect.models.resnet import ResNet18

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='USTC Computer Vision Final Project')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_path', default='datasets/fer2013/fer2013.csv', type=str)
parser.add_argument('--checkpoint', default='models/expression_detect/best_checkpoint.tar', type=str)
parser.add_argument('--arch', default="ResNet18", type=str)
parser.add_argument('--Ncrop', default=True, type=eval)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)



class ExpressionDetect:
    def __init__(self, bbox_net):
        args = parser.parse_args()
        self.model = ResNet18().to(device)
        checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.bbox_net = bbox_net


    def get_faces(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
        frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.bbox_net.setInput(blob)
        detections = self.bbox_net.forward()  # 网络进行前向传播，检测人脸
        bboxs = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxs.append([x1, y1, x2, y2])  # bounding box 的坐标
        for bbox in bboxs:
            bbox[0] = max(0, min(frame.shape[1], bbox[0]))
            bbox[1] = max(0, min(frame.shape[0], bbox[1]))
            bbox[2] = max(0, min(frame.shape[1], bbox[2]))
            bbox[3] = max(0, min(frame.shape[0], bbox[3]))
        return bboxs

    def img_process(self, img, device):
        mu, st = 0, 255

        emo = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']
        test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))
        ])
        img = Image.fromarray(img)
        img = test_transform(img)
        ncrops, c, h, w = img.shape
        inputs = img.view(-1, c, h, w).to(device)
        # forward
        outputs = self.model(inputs)
        outputs = outputs.view(1, ncrops, -1)
        outputs = torch.sum(outputs, dim=1) / ncrops
        maxk = max((1, 2))

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        emotion = ''
        if pred.data[0][0] == 4:
            emotion = 'sad'
        if pred.data[0][0] == 3 or pred.data[0][0] == 5:
            emotion = 'happy'
        if pred.data[0][0] == 6 or pred.data[0][0] == 0 or pred.data[0][0] == 1 or pred.data[0][0] == 2:
            emotion = 'normal'
        return emotion

    def process(self, frame, keypressed):
        frame = cv2.flip(frame, 1)
        bboxs = self.get_faces(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for bbox in bboxs:
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            emotion = self.img_process(cv2.resize(gray[bbox[1]:bbox[3], bbox[0]:bbox[2]], [48, 48]), device)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            frame = cv2.putText(frame, emotion, (bbox[0], bbox[1]), font, 2.0, (0, 255, 255), 1)
        frame = cv2.flip(frame, 1)
        return frame

