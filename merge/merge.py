# -*-coding:utf-8-*-
import os

import cv2
from tqdm import tqdm


def get_name(filename):
    for i in range(len(filename)):
        if filename[i] == '_':
            return filename[:i]
    return filename


def merge(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError(img1.shape, img2.shape)
    img = img1.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j, 0] = max(img1[i, j, 0], img2[i, j, 0])
            img[i, j, 1] = max(img1[i, j, 1], img2[i, j, 1])
            img[i, j, 2] = max(img1[i, j, 2], img2[i, j, 2])
    return img


def merge_imgs(parent_dir):
    filenames = os.listdir(parent_dir)
    img_set = set(map(get_name, filenames))
    img_dict = {}
    for name in img_set:
        img_dict[name] = None
    for filename in tqdm(filenames):
        img = cv2.imread(os.path.join(parent_dir, filename))
        img = cv2.resize(img, (640, 480))
        name = get_name(filename)
        if img_dict.get(name) is None:
            img_dict[name] = img.copy()
        else:
            img_dict[name] = merge(img_dict.get(name), img)

    for name in img_dict:
        cv2.imwrite(os.path.join('result', name + '.bmp'), img_dict[name])


if __name__ == '__main__':
    merge_imgs('2/run_dns/final_imsk')
    merge_imgs('3/run_dns/final_imsk')
