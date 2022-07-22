import math
import time

import cv2  # pip install opencv-python
import numpy as np
# mdeiapipe 不能使用conda装  只能用pip装     装之前最好换一下pip源
# 导入mediapipe：https://google.github.io/mediapipe/solutions/hands
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 获取摄像头视频流
cap = cv2.VideoCapture(0)

# 界面方块的参数
square_x = 100
square_y = 100
square_width = 100

# 获取画面的宽高,用于恢复手指在图片上的坐标
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 方块初始数组
x = 100
y = 100
w = 200
h = 200

L1 = 0
L2 = 0

on_square = False
square_color = (0, 255, 0)


class Game:
    def __init__(self):
        self.points = []
        self.score_grade = [20, 40, 60, 80]
        self.speeds = [10, 15, 20, 30]
        self.start_time = time.time()
        self.death = False
        self.score = 0
        self.press_time = None

    def create(self, frame, max_count=2):
        for i in range(np.random.randint(0, max_count)):
            self.points.append([int(np.random.random() * frame.shape[1]), 0])

    def update(self, frame, index_x, index_y, middle_x, middle_y, thresh=10):
        if not self.death:
            self.create(frame)
            speed = self.speeds[0]
            self.score = time.time() - self.start_time
            n = len(self.score_grade)
            for i in range(n):
                grade = self.score_grade[n - i - 1]
                if self.score > grade:
                    speed = self.speeds[n - i - 1]
                    break
            for i in range(len(self.points)):
                point = self.points[len(self.points) - i - 1]
                point[1] += speed
                if point[1] >= frame.shape[0]:
                    self.points.remove(point)
                elif np.sqrt(np.power(point[0] - index_x, 2) + np.power(point[1] - index_y, 2)) < thresh:
                    self.points.clear()
                    self.death = True
                    break
                else:
                    cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv2.putText(frame, 'Time:' + str(np.round(self.score, 2)) + ' s', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'Your are dead !', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, 'Time:' + str(np.round(self.score, 2)) + ' s', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if np.sqrt(np.power(middle_x - index_x, 2) + np.power(middle_y - index_y, 2)) < thresh * 2:
                if self.press_time is None:
                    self.press_time = time.time()
            else:
                if self.press_time is not None and time.time() - self.press_time > 2:
                    self.death = False
                    self.start_time = time.time()
                    self.death = False
                    self.score = 0
                self.press_time = None
        return frame


game = Game()

while True:

    # 读取每一帧  ret(bool) :  代表是否打开成功摄像头   frame （numpy.ndarray） ： 单帧的图像
    # tips：opencv的读取是BGR的顺序，很多算法是RGB，所以需要转化。
    ret, frame = cap.read()
    # print(type(frame))
    # print(type(ret))
    if not ret:
        print("无法打开摄像头")
        continue

    # print(ret)
    # 对图像进行处理，镜像一下，围绕y轴
    frame = cv2.flip(frame, 1)

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 识别
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 判断是否出现手
    if results.multi_hand_landmarks:
        # 解析便利每一双手
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制21个关键点
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            """
            print(hand_landmarks)
            每个关键点的解析 
            landmark {
                  x: 0.18473060429096222
                  y: 0.058572977781295776
                  z: -0.10718432068824768
                }
            """
            # 21 个关键点的x,y坐标列表
            x_list = []
            y_list = []
            for landmark in hand_landmarks.landmark:
                x_list.append(landmark.x)
                y_list.append(landmark.y)

            # 输出一下长度，21 就识别全了
            # print(len(x_list))

            # 获取食指指尖坐标，坐标位置查看：https://google.github.io/mediapipe/solutions/hands
            index_finger_x = int(x_list[8] * width)
            index_finger_y = int(y_list[8] * height)
            # 食指尖画圆
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (255, 0, 255), -1)
            # 获取中指坐标
            middle_finger_x = int(x_list[4] * width)
            middle_finger_y = int(y_list[4] * height)
            cv2.circle(frame, (middle_finger_x, middle_finger_y), 10, (255, 0, 255), -1)

            # 计算两指距离
            # finger_distance =math.sqrt( (middle_finger_x - index_finger_x)**2 + (middle_finger_y-index_finger_y)**2)
            # finger_distance = math.hypot((middle_finger_x - index_finger_x), (middle_finger_y - index_finger_y))
            # 判断食指指尖在不在方块上
            frame = game.update(frame, index_finger_x, index_finger_y, middle_finger_x, middle_finger_y)
    else:
        frame = game.update(frame, np.inf, np.inf, 0, 0)
    # 画一个正方形，需要实心
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-1)

    # 半透明处理
    # overlay = frame.copy()
    # cv2.rectangle(frame, (x, y), (x + w, y + h), square_color, -1)
    # frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)

    # 看一下距离
    # print(finger_distance)
    # 此时图片是BGR 不是RGB     -1  代表实心      255 = b  0 = g 0 =r    所以是蓝色方块
    # cv2.rectangle(frame, (square_x, square_y), (square_x + square_width, square_y + square_width), (255, 0, 0), -1)

    # 显示
    cv2.imshow('Virtual drag', frame)

    # 退出条件 esc 退出
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
