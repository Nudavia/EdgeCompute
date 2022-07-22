# -*-coding:utf-8-*-
import socket

import cv2
import numpy as np


class Server:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定端口
        self.server.bind(('192.168.43.55', 8887))
        self.keypressed = 255
        # self.server.setblocking(0)  # 设置为非阻塞模式

    def run(self):
        while True:
            try:
                data, addr = self.server.recvfrom(921600)
                print("接收信息的来源： %s:%s" % addr)
                # print("接收信息的数据： %s" % data.decode('utf-8'))
                receive_data = np.frombuffer(data, dtype='uint8')
                img = cv2.imdecode(receive_data, 1)
                # cv2.putText(img, "Server", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                cv2.imshow('Server', img)
                self.keypressed = cv2.waitKey(1) & 0xFF
                self.server.sendto(str(self.keypressed).encode('utf-8'), addr)
                # 退出系统操作
                if self.keypressed == 27:
                    break
            except BlockingIOError as e:
                print(e)
        self.server.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    server = Server()
    server.run()
