# -*-coding:utf-8-*-
import os.path

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For static images:
DIR = 'background'
IMAGE_FILES = ['bg.png', 'bg2.jpg', 'bg3.jpg', 'bg4.png', 'bg5.jpg', 'bg6.jpg', 'bg7.jpg']
BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

# For webcam input:
BG_COLOR = (192, 192, 192)  # gray




class Segment:
    def __init__(self):
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1)
        self.bgs = []
        self.bg_index = 0
        for img in IMAGE_FILES:
            self.bgs.append(cv2.imread(os.path.join(DIR, img)))

    def process(self, frame, keypressed):
        if keypressed == ord('g'):
            self.bg_index = (self.bg_index + 1) % len(self.bgs)

        bg_image = cv2.resize(self.bgs[self.bg_index], (frame.shape[1], frame.shape[0]))
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        return output_image
