import cv2
import os
import time
import torch
import argparse



def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


from .nanodet.model.arch import build_model
from .nanodet.util import load_model_weight
from .nanodet.data.transform import Pipeline

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
'''目标检测-图片'''
# python object_detect.py image --config ./config/nanodet-m.yml --model model/nanodet_m.pth --path  street.png

'''目标检测-视频文件'''
# python object_detect.py video --config ./config/nanodet-m.yml --model model/nanodet_m.pth --path  test.mp4

'''目标检测-摄像头'''


# python object_detect.py webcam --config ./config/nanodet-m.yml --model model/nanodet_m.pth --path  0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='webcam', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', help='model config file path', default='./config/nanodet-m.yml')
    parser.add_argument('--model', help='model file path', default='model/nanodet_m.pth')
    parser.add_argument('--path', default='./demo', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        print('viz time: {:.3f}s'.format(time.time() - time1))
        return self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


args = parse_args()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
load_config(cfg, args.config)
logger = Logger(-1, use_tensorboard=False)


class ObjectDetect:
    def __init__(self):
        self.predictor = Predictor(cfg, args.model, logger, device='cuda:0')

    def process(self, frame, keypressed):
        meta, res = self.predictor.inference(frame)
        return self.predictor.visualize(res, meta, cfg.class_names, 0.35)
